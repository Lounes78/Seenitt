"""Stage 3: Intra-Track Filtering - Select best crop from each tracking ID."""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import shutil

from torchmetrics.multimodal import CLIPImageQualityAssessment
from .base_stage import BaseStage
from ..utils import ensure_dir


class IntraTrackFilterStage(BaseStage):
    """Select the best quality crop for each tracking ID."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize intra-track filtering stage.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "intra_track_filter")
        
        # Quality assessment configuration
        quality_config = config.get('quality', {})
        self.clip_model_name = quality_config.get('clip_model', 'openai/clip-vit-base-patch32')
        self.image_size = quality_config.get('image_size', [224, 224])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Selection criteria weights
        self.quality_weight = quality_config.get('quality_weight', 0.4)
        self.confidence_weight = quality_config.get('confidence_weight', 0.3)
        self.area_weight = quality_config.get('area_weight', 0.2)
        self.centrality_weight = quality_config.get('centrality_weight', 0.1)
        
        # Initialize quality assessment model
        self.quality_metric = None
        self._load_quality_model()
        
        self.processed_tracks = 0
        self.total_crops = 0
        self.selected_crops = 0
    
    def _load_quality_model(self) -> None:
        """Load CLIP quality assessment model."""
        try:
            self.quality_metric = CLIPImageQualityAssessment().to(self.device)
            self.logger.info(f"CLIP quality model loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP quality model: {e}")
            raise
    
    def _extract_track_id_from_filename(self, filename: str) -> Optional[int]:
        """Extract track ID from filename.
        
        Args:
            filename: Image filename
            
        Returns:
            Track ID if found, None otherwise
        """
        try:
            # Expected format: class_{class_id}_track_{track_id}_frame_{frame}_bbox_{idx}.jpg
            parts = filename.split('_')
            if len(parts) >= 4 and parts[2] == 'track':
                return int(parts[3])
        except (ValueError, IndexError):
            pass
        return None
    
    def _calculate_image_quality_score(self, image_path: str) -> float:
        """Calculate image quality score using CLIP.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img_resized = cv2.resize(img_rgb, tuple(self.image_size))
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(img_resized).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Calculate quality score
            with torch.no_grad():
                score = self.quality_metric(img_tensor).item()
            
            # Clean up GPU memory
            del img_tensor
            torch.cuda.empty_cache()
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality for {image_path}: {e}")
            return 0.0
    
    def _calculate_centrality_score(self, bbox: List[float], image_shape: Tuple[int, int]) -> float:
        """Calculate how centered the bounding box is in the image.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_shape: Image dimensions (height, width)
            
        Returns:
            Centrality score between 0 and 1 (1 = perfectly centered)
        """
        if not bbox or len(bbox) != 4:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_shape
        
        # Calculate bbox center
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Calculate image center
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        
        # Calculate distance from center (normalized)
        distance_x = abs(bbox_center_x - img_center_x) / (img_w / 2)
        distance_y = abs(bbox_center_y - img_center_y) / (img_h / 2)
        
        # Calculate centrality score (1 - normalized distance)
        centrality = 1 - min(1.0, (distance_x + distance_y) / 2)
        return centrality
    
    def _calculate_area_score(self, bbox: List[float], image_shape: Tuple[int, int]) -> float:
        """Calculate area score (larger objects are generally better).
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_shape: Image dimensions (height, width)
            
        Returns:
            Area score between 0 and 1
        """
        if not bbox or len(bbox) != 4:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_shape
        
        # Calculate bbox area
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Calculate total image area
        total_area = img_h * img_w
        
        # Normalize area (but cap at 0.5 to avoid preferring too large objects)
        area_ratio = min(0.5, bbox_area / total_area)
        return area_ratio * 2  # Scale to 0-1 range
    
    def _calculate_composite_score(self, crop_info: Dict[str, Any], 
                                 quality_score: float, image_shape: Tuple[int, int]) -> float:
        """Calculate composite score for crop selection.
        
        Args:
            crop_info: Crop information dictionary
            quality_score: CLIP quality score
            image_shape: Image dimensions
            
        Returns:
            Composite score for ranking
        """
        # Extract features
        confidence = crop_info.get('confidence', 0.0)
        bbox = crop_info.get('bbox', [])
        
        # Calculate individual scores
        centrality_score = self._calculate_centrality_score(bbox, image_shape)
        area_score = self._calculate_area_score(bbox, image_shape)
        
        # Calculate weighted composite score
        composite_score = (
            self.quality_weight * quality_score +
            self.confidence_weight * confidence +
            self.centrality_weight * centrality_score +
            self.area_weight * area_score
        )
        
        return composite_score
    
    def _group_crops_by_track(self, filtered_crops: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group crops by track ID.
        
        Args:
            filtered_crops: List of filtered crop information
            
        Returns:
            Dictionary mapping track IDs to crop lists
        """
        track_groups = defaultdict(list)
        
        for crop_info in filtered_crops:
            track_id = crop_info.get('track_id')
            if track_id is not None:
                track_groups[track_id].append(crop_info)
        
        return track_groups
    
    def _select_best_crop_for_track(self, track_crops: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best crop for a single track.
        
        Args:
            track_crops: List of crops for this track
            
        Returns:
            Best crop information with scores
        """
        if len(track_crops) == 1:
            # Only one crop, calculate its scores and return
            crop = track_crops[0]
            image_path = crop.get('filtered_filepath', crop.get('filepath'))
            
            if image_path and Path(image_path).exists():
                quality_score = self._calculate_image_quality_score(image_path)
                
                # Load image to get shape
                img = cv2.imread(image_path)
                image_shape = img.shape[:2] if img is not None else (224, 224)
                
                composite_score = self._calculate_composite_score(crop, quality_score, image_shape)
                
                crop_with_scores = crop.copy()
                crop_with_scores.update({
                    'quality_score': quality_score,
                    'composite_score': composite_score,
                    'selection_reason': 'only_crop_for_track'
                })
                
                return crop_with_scores
        
        # Multiple crops - evaluate each and select best
        crop_scores = []
        
        for crop in track_crops:
            image_path = crop.get('filtered_filepath', crop.get('filepath'))
            
            if not image_path or not Path(image_path).exists():
                continue
            
            # Calculate quality score
            quality_score = self._calculate_image_quality_score(image_path)
            
            # Load image to get shape
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            image_shape = img.shape[:2]
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(crop, quality_score, image_shape)
            
            crop_with_scores = crop.copy()
            crop_with_scores.update({
                'quality_score': quality_score,
                'composite_score': composite_score
            })
            
            crop_scores.append(crop_with_scores)
        
        if not crop_scores:
            return None
        
        # Select crop with highest composite score
        best_crop = max(crop_scores, key=lambda x: x['composite_score'])
        best_crop['selection_reason'] = f'best_of_{len(crop_scores)}_crops'
        
        return best_crop
    
    def process(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Main processing method for intra-track filtering.
        
        Args:
            input_data: Results from tracking ID filtering stage
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing intra-track filtering results
        """
        self.logger.info("Starting intra-track filtering (best crop selection)")
        
        filtered_crop_results = input_data['filtered_crop_results']
        filtered_crops = filtered_crop_results['filtered_crops']
        
        # Group crops by track ID
        track_groups = self._group_crops_by_track(filtered_crops)
        self.processed_tracks = len(track_groups)
        self.total_crops = len(filtered_crops)
        
        # Create output directory for selected crops
        selected_dir = Path(output_dir) / "selected_crops"
        ensure_dir(selected_dir)
        
        # Select best crop for each track
        selected_crops = []
        track_selection_info = {}
        
        for track_id, crops in track_groups.items():
            self.logger.debug(f"Processing track {track_id} with {len(crops)} crops")
            
            best_crop = self._select_best_crop_for_track(crops)
            
            if best_crop:
                # Copy selected crop to output directory
                original_path = Path(best_crop.get('filtered_filepath', best_crop.get('filepath')))\n                new_filename = f"track_{track_id}_selected.jpg"
                new_path = selected_dir / new_filename
                
                try:
                    shutil.copy2(original_path, new_path)
                    
                    # Update crop info
                    best_crop['selected_filepath'] = str(new_path)
                    best_crop['selected_filename'] = new_filename
                    selected_crops.append(best_crop)
                    self.selected_crops += 1
                    
                    # Store selection info
                    track_selection_info[track_id] = {
                        'total_crops': len(crops),
                        'selected_crop': best_crop['filename'],
                        'quality_score': best_crop['quality_score'],
                        'composite_score': best_crop['composite_score'],
                        'selection_reason': best_crop['selection_reason']
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to copy selected crop for track {track_id}: {e}")
        
        # Compile results
        results = {
            'selected_crops': selected_crops,
            'track_selection_info': track_selection_info,
            'selected_directory': str(selected_dir),
            'selection_criteria': {
                'quality_weight': self.quality_weight,
                'confidence_weight': self.confidence_weight,
                'centrality_weight': self.centrality_weight,
                'area_weight': self.area_weight
            },
            'stage_metrics': self.get_metrics()
        }
        
        # Save results
        self.save_results(results, output_dir)
        
        self.logger.info(f"Selected {self.selected_crops} crops from {self.processed_tracks} tracks "
                        f"(total {self.total_crops} crops)")
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for this stage.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'processed_tracks': self.processed_tracks,
            'total_input_crops': self.total_crops,
            'selected_crops': self.selected_crops,
            'selection_rate': self.selected_crops / max(1, self.total_crops),
            'avg_crops_per_track': self.total_crops / max(1, self.processed_tracks)
        }