"""Stage 4: Similarity Detection - Remove duplicate detections using embeddings."""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import shutil
from collections import defaultdict

from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from .base_stage import BaseStage
from ..utils import ensure_dir


class SimilarityDetectionStage(BaseStage):
    """Remove duplicate detections across different tracking IDs using embedding similarity."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize similarity detection stage.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "similarity_detection")
        
        # Similarity configuration
        similarity_config = config.get('similarity', {})
        self.embedding_model_name = similarity_config.get('embedding_model', 'openai/clip-vit-base-patch32')
        self.similarity_threshold = similarity_config.get('threshold', 0.85)
        self.batch_size = similarity_config.get('batch_size', 16)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Advanced similarity options
        self.use_spatial_constraint = similarity_config.get('use_spatial_constraint', True)
        self.spatial_distance_threshold = similarity_config.get('spatial_distance_threshold', 0.3)
        self.temporal_distance_threshold = similarity_config.get('temporal_distance_threshold', 30)  # frames
        
        # Initialize CLIP model
        self.clip_model = None
        self.clip_processor = None
        self._load_models()
        
        self.processed_images = 0
        self.similar_groups = []
        self.unique_images = 0
    
    def _load_models(self) -> None:
        """Load CLIP model and processor."""
        try:
            self.clip_model = CLIPModel.from_pretrained(self.embedding_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.embedding_model_name)
            self.logger.info(f"CLIP model loaded: {self.embedding_model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Extract CLIP embeddings for a list of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        # Process images in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            
            # Load batch images
            for img_path in batch_paths:
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        batch_images.append(img_rgb)
                    else:
                        # Add placeholder for failed images
                        batch_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
                        self.logger.warning(f"Failed to load image: {img_path}")
                except Exception as e:
                    self.logger.warning(f"Error loading {img_path}: {e}")
                    batch_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            if batch_images:
                try:
                    # Process batch
                    inputs = self.clip_processor(images=batch_images, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        features = self.clip_model.get_image_features(**inputs)
                        # Normalize embeddings
                        features = features / features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy and add to embeddings list
                    batch_embeddings = features.cpu().numpy()
                    embeddings.append(batch_embeddings)
                    
                    # Clean up GPU memory
                    del features, inputs
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch: {e}")
                    # Add zero embeddings for failed batch
                    dummy_embeddings = np.zeros((len(batch_images), 512))
                    embeddings.append(dummy_embeddings)
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])
    
    def _calculate_spatial_distance(self, crop1: Dict[str, Any], crop2: Dict[str, Any]) -> float:
        """Calculate normalized spatial distance between two crops.
        
        Args:
            crop1: First crop information
            crop2: Second crop information
            
        Returns:
            Normalized spatial distance (0 = same location, 1 = far apart)
        """
        bbox1 = crop1.get('bbox', [0, 0, 100, 100])
        bbox2 = crop2.get('bbox', [0, 0, 100, 100])
        
        # Calculate centers
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        # Calculate distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # Normalize by image diagonal (assume 1920x1080 for normalization)
        max_distance = np.sqrt(1920**2 + 1080**2)
        normalized_distance = min(1.0, distance / max_distance)
        
        return normalized_distance
    
    def _calculate_temporal_distance(self, crop1: Dict[str, Any], crop2: Dict[str, Any]) -> int:
        """Calculate temporal distance between two crops.
        
        Args:
            crop1: First crop information
            crop2: Second crop information
            
        Returns:
            Frame difference between crops
        """
        frame1 = crop1.get('frame', 0)
        frame2 = crop2.get('frame', 0)
        return abs(frame1 - frame2)
    
    def _should_compare_crops(self, crop1: Dict[str, Any], crop2: Dict[str, Any]) -> bool:
        """Determine if two crops should be compared for similarity.
        
        Args:
            crop1: First crop information
            crop2: Second crop information
            
        Returns:
            True if crops should be compared, False otherwise
        """
        # Always compare crops from different tracks
        if crop1.get('track_id') == crop2.get('track_id'):
            return False
        
        # Apply spatial constraint if enabled
        if self.use_spatial_constraint:
            spatial_dist = self._calculate_spatial_distance(crop1, crop2)
            if spatial_dist > self.spatial_distance_threshold:
                return False
        
        # Apply temporal constraint
        temporal_dist = self._calculate_temporal_distance(crop1, crop2)
        if temporal_dist > self.temporal_distance_threshold:
            return False
        
        return True
    
    def _find_similar_groups(self, selected_crops: List[Dict[str, Any]], 
                           embeddings: np.ndarray) -> List[List[int]]:
        """Find groups of similar images.
        
        Args:
            selected_crops: List of selected crop information
            embeddings: Corresponding embeddings array
            
        Returns:
            List of similar groups (each group is a list of indices)
        """
        if len(embeddings) == 0:
            return []
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        similar_groups = []
        processed_indices = set()
        
        for i in range(len(similarity_matrix)):
            if i in processed_indices:
                continue
            
            current_group = [i]
            processed_indices.add(i)
            
            for j in range(i + 1, len(similarity_matrix)):
                if j in processed_indices:
                    continue
                
                # Check if images should be compared
                if not self._should_compare_crops(selected_crops[i], selected_crops[j]):
                    continue
                
                # Check similarity threshold
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    current_group.append(j)
                    processed_indices.add(j)
            
            # Only consider groups with multiple items as similar groups
            if len(current_group) > 1:
                similar_groups.append(current_group)
        
        return similar_groups
    
    def _select_best_from_group(self, group_indices: List[int], 
                               selected_crops: List[Dict[str, Any]]) -> int:
        """Select the best image from a similar group.
        
        Args:
            group_indices: List of indices in the similar group
            selected_crops: List of crop information
            
        Returns:
            Index of the best image in the group
        """
        best_idx = group_indices[0]
        best_score = -1
        
        for idx in group_indices:
            crop = selected_crops[idx]
            
            # Calculate selection score based on multiple criteria
            quality_score = crop.get('quality_score', 0.0)
            composite_score = crop.get('composite_score', 0.0)
            confidence = crop.get('confidence', 0.0)
            
            # Combine scores (you can adjust weights as needed)
            combined_score = (
                0.5 * quality_score +
                0.3 * composite_score +
                0.2 * confidence
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx
        
        return best_idx
    
    def process(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Main processing method for similarity detection.
        
        Args:
            input_data: Results from intra-track filtering stage
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing similarity detection results
        """
        self.logger.info("Starting similarity detection for duplicate removal")
        
        selected_crops = input_data['selected_crops']
        self.processed_images = len(selected_crops)
        
        if self.processed_images == 0:
            self.logger.warning("No selected crops to process")
            return {
                'unique_crops': [],
                'similar_groups': [],
                'unique_directory': '',
                'stage_metrics': self.get_metrics()
            }
        
        # Extract image paths
        image_paths = []
        for crop in selected_crops:
            img_path = crop.get('selected_filepath', crop.get('filtered_filepath', crop.get('filepath')))
            image_paths.append(img_path)
        
        self.logger.info(f"Extracting embeddings for {len(image_paths)} images")
        
        # Extract embeddings
        embeddings = self._extract_embeddings(image_paths)
        
        if len(embeddings) == 0:
            self.logger.error("Failed to extract embeddings")
            return {
                'unique_crops': [],
                'similar_groups': [],
                'unique_directory': '',
                'stage_metrics': self.get_metrics()
            }
        
        # Find similar groups
        self.logger.info("Finding similar groups")
        similar_groups_indices = self._find_similar_groups(selected_crops, embeddings)
        self.similar_groups = similar_groups_indices
        
        # Create output directory for unique crops
        unique_dir = Path(output_dir) / "unique_crops"
        ensure_dir(unique_dir)
        
        # Track which images to keep
        indices_to_remove = set()
        group_info = []
        
        # Process each similar group
        for group_idx, group_indices in enumerate(similar_groups_indices):
            best_idx = self._select_best_from_group(group_indices, selected_crops)
            
            # Mark others for removal
            for idx in group_indices:
                if idx != best_idx:
                    indices_to_remove.add(idx)
            
            # Store group information
            group_crops = [selected_crops[idx] for idx in group_indices]
            group_info.append({
                'group_id': group_idx + 1,
                'size': len(group_indices),
                'selected_index': best_idx,
                'selected_track_id': selected_crops[best_idx].get('track_id'),
                'removed_track_ids': [selected_crops[idx].get('track_id') for idx in group_indices if idx != best_idx],
                'similarity_scores': [
                    float(cosine_similarity([embeddings[best_idx]], [embeddings[idx]])[0][0])
                    for idx in group_indices if idx != best_idx
                ]
            })
        
        # Copy unique images to output directory
        unique_crops = []
        for idx, crop in enumerate(selected_crops):
            if idx not in indices_to_remove:
                # Copy image to unique directory
                original_path = Path(crop.get('selected_filepath', 
                                           crop.get('filtered_filepath', 
                                                   crop.get('filepath'))))
                
                new_filename = f"unique_track_{crop.get('track_id')}.jpg"
                new_path = unique_dir / new_filename
                
                try:
                    shutil.copy2(original_path, new_path)
                    
                    # Update crop info
                    unique_crop = crop.copy()
                    unique_crop['unique_filepath'] = str(new_path)
                    unique_crop['unique_filename'] = new_filename
                    unique_crops.append(unique_crop)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to copy unique crop {original_path}: {e}")
        
        self.unique_images = len(unique_crops)
        
        # Compile results
        results = {
            'unique_crops': unique_crops,
            'similar_groups': group_info,
            'unique_directory': str(unique_dir),
            'similarity_config': {
                'threshold': self.similarity_threshold,
                'use_spatial_constraint': self.use_spatial_constraint,
                'spatial_distance_threshold': self.spatial_distance_threshold,
                'temporal_distance_threshold': self.temporal_distance_threshold
            },
            'stage_metrics': self.get_metrics()
        }
        
        # Save results
        self.save_results(results, output_dir)
        
        self.logger.info(f"Similarity detection complete: {self.unique_images} unique images "
                        f"from {self.processed_images} input images "
                        f"({len(similar_groups_indices)} similar groups found)")
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for this stage.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'input_images': self.processed_images,
            'unique_images': self.unique_images,
            'similar_groups_found': len(self.similar_groups),
            'duplicate_removal_rate': (self.processed_images - self.unique_images) / max(1, self.processed_images),
            'similarity_threshold': self.similarity_threshold
        }