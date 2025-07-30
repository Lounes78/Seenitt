"""Stage 2: Tracking ID Filtering - Filter tracking IDs based on criteria."""

import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Any, List, Set

from .base_stage import BaseStage
from ..utils import ensure_dir


class TrackingIDFilterStage(BaseStage):
    """Filter tracking IDs based on various criteria."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tracking ID filter stage.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "tracking_id_filter")
        
        # Filtering criteria
        filtering_config = config.get('filtering', {})
        self.min_detections_per_track = filtering_config.get('min_detections_per_track', 3)
        self.min_track_duration = filtering_config.get('min_track_duration', 1.0)  # seconds
        self.min_confidence_avg = filtering_config.get('min_confidence_avg', 0.3)
        self.max_confidence_std = filtering_config.get('max_confidence_std', 0.3)
        self.min_bbox_consistency = filtering_config.get('min_bbox_consistency', 0.7)
        
        self.track_stats = {}
        self.filtered_tracks = set()
        self.total_tracks = 0
        
    def _calculate_track_statistics(self, detection_results: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Calculate statistics for each tracking ID.
        
        Args:
            detection_results: Results from detection stage
            
        Returns:
            Dictionary with track statistics
        """
        track_data = defaultdict(list)
        
        # Collect all detections for each track
        for frame_data in detection_results['frames']:
            frame_num = frame_data['frame']
            timestamp = frame_data.get('timestamp', frame_num / 30.0)
            
            for obj in frame_data['objects']:
                track_id = obj.get('track_id')
                if track_id is not None:
                    track_data[track_id].append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'confidence': obj['confidence'],
                        'bbox': obj['bbox'],
                        'bbox_area': obj.get('bbox_area', 0),
                        'class_id': obj['class_id']
                    })
        
        # Calculate statistics for each track
        track_stats = {}
        
        for track_id, detections in track_data.items():
            if not detections:
                continue
                
            # Basic counts and timing
            num_detections = len(detections)
            timestamps = [d['timestamp'] for d in detections]
            duration = max(timestamps) - min(timestamps)
            
            # Confidence statistics
            confidences = [d['confidence'] for d in detections]
            confidence_avg = sum(confidences) / len(confidences)
            confidence_std = self._calculate_std(confidences, confidence_avg)
            
            # Bbox consistency (measure of tracking stability)
            bbox_consistency = self._calculate_bbox_consistency(detections)
            
            # Class consistency
            class_ids = [d['class_id'] for d in detections]
            class_counter = Counter(class_ids)
            most_common_class = class_counter.most_common(1)[0][0]
            class_consistency = class_counter[most_common_class] / len(class_ids)
            
            # Area statistics
            areas = [d['bbox_area'] for d in detections]
            area_avg = sum(areas) / len(areas)
            area_std = self._calculate_std(areas, area_avg)
            
            track_stats[track_id] = {
                'num_detections': num_detections,
                'duration': duration,
                'confidence_avg': confidence_avg,
                'confidence_std': confidence_std,
                'bbox_consistency': bbox_consistency,
                'class_consistency': class_consistency,
                'most_common_class': most_common_class,
                'area_avg': area_avg,
                'area_std': area_std,
                'first_frame': min(d['frame'] for d in detections),
                'last_frame': max(d['frame'] for d in detections),
                'detections': detections
            }\n        
        return track_stats
    
    def _calculate_std(self, values: List[float], mean: float) -> float:
        """Calculate standard deviation.
        
        Args:
            values: List of values
            mean: Mean of the values
            
        Returns:
            Standard deviation
        """
        if len(values) <= 1:
            return 0.0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_bbox_consistency(self, detections: List[Dict[str, Any]]) -> float:
        """Calculate bounding box consistency score.
        
        Args:
            detections: List of detection objects
            
        Returns:
            Consistency score between 0 and 1
        """
        if len(detections) <= 1:
            return 1.0
        
        # Calculate IoU between consecutive bounding boxes
        ious = []
        for i in range(1, len(detections)):
            bbox1 = detections[i-1]['bbox']
            bbox2 = detections[i]['bbox']
            iou = self._calculate_iou(bbox1, bbox2)
            ious.append(iou)
        
        # Return average IoU as consistency measure
        return sum(ious) / len(ious) if ious else 0.0
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _apply_filtering_criteria(self, track_stats: Dict[int, Dict[str, Any]]) -> Set[int]:
        """Apply filtering criteria to select valid tracks.
        
        Args:
            track_stats: Track statistics dictionary
            
        Returns:
            Set of valid track IDs
        """
        valid_tracks = set()
        
        for track_id, stats in track_stats.items():
            # Check minimum detections per track
            if stats['num_detections'] < self.min_detections_per_track:
                self.logger.debug(f"Track {track_id} filtered: insufficient detections "
                                f"({stats['num_detections']} < {self.min_detections_per_track})")
                continue
            
            # Check minimum track duration
            if stats['duration'] < self.min_track_duration:
                self.logger.debug(f"Track {track_id} filtered: insufficient duration "
                                f"({stats['duration']:.2f}s < {self.min_track_duration}s)")
                continue
            
            # Check average confidence
            if stats['confidence_avg'] < self.min_confidence_avg:
                self.logger.debug(f"Track {track_id} filtered: low average confidence "
                                f"({stats['confidence_avg']:.3f} < {self.min_confidence_avg})")
                continue
            
            # Check confidence consistency (low std deviation is better)
            if stats['confidence_std'] > self.max_confidence_std:
                self.logger.debug(f"Track {track_id} filtered: high confidence variance "
                                f"({stats['confidence_std']:.3f} > {self.max_confidence_std})")
                continue
            
            # Check bbox consistency
            if stats['bbox_consistency'] < self.min_bbox_consistency:
                self.logger.debug(f"Track {track_id} filtered: low bbox consistency "
                                f"({stats['bbox_consistency']:.3f} < {self.min_bbox_consistency})")
                continue
            
            # Track passed all criteria
            valid_tracks.add(track_id)
            self.logger.debug(f"Track {track_id} passed all filtering criteria")
        
        return valid_tracks
    
    def filter_crops_by_tracks(self, crop_results: Dict[str, Any], 
                              valid_tracks: Set[int], output_dir: str) -> Dict[str, Any]:
        """Filter cropped images based on valid track IDs.
        
        Args:
            crop_results: Results from cropping stage
            valid_tracks: Set of valid track IDs
            output_dir: Output directory for filtered crops
            
        Returns:
            Dictionary with filtered crop results
        """
        filtered_dir = Path(output_dir) / "filtered_crops"
        ensure_dir(filtered_dir)
        
        filtered_crops = []
        total_filtered = 0
        
        for crop_info in crop_results['cropped_images']:
            track_id = crop_info.get('track_id')
            
            if track_id in valid_tracks:
                # Copy file to filtered directory
                original_path = Path(crop_info['filepath'])
                new_path = filtered_dir / original_path.name
                
                try:
                    import shutil
                    shutil.copy2(original_path, new_path)
                    
                    # Update crop info with new path
                    filtered_crop_info = crop_info.copy()
                    filtered_crop_info['filtered_filepath'] = str(new_path)
                    filtered_crops.append(filtered_crop_info)
                    total_filtered += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to copy {original_path}: {e}")
        
        filtered_results = {
            'total_filtered_crops': total_filtered,
            'filtered_directory': str(filtered_dir),
            'filtered_crops': filtered_crops,
            'filter_rate': total_filtered / max(1, len(crop_results['cropped_images']))
        }
        
        self.logger.info(f"Filtered {total_filtered} crops from {len(crop_results['cropped_images'])} total")
        return filtered_results
    
    def process(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Main processing method for tracking ID filtering.
        
        Args:
            input_data: Results from detection stage
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing filtering results
        """
        self.logger.info("Starting tracking ID filtering")
        
        detection_results = input_data['detection_results']
        crop_results = input_data['crop_results']
        
        # Calculate track statistics
        self.track_stats = self._calculate_track_statistics(detection_results)
        self.total_tracks = len(self.track_stats)
        
        # Apply filtering criteria
        valid_tracks = self._apply_filtering_criteria(self.track_stats)
        self.filtered_tracks = valid_tracks
        
        # Filter cropped images
        filtered_crop_results = self.filter_crops_by_tracks(crop_results, valid_tracks, output_dir)
        
        # Compile results
        results = {
            'track_statistics': self.track_stats,
            'valid_tracks': list(valid_tracks),
            'filtered_crop_results': filtered_crop_results,
            'filtering_criteria': {
                'min_detections_per_track': self.min_detections_per_track,
                'min_track_duration': self.min_track_duration,
                'min_confidence_avg': self.min_confidence_avg,
                'max_confidence_std': self.max_confidence_std,
                'min_bbox_consistency': self.min_bbox_consistency
            },
            'stage_metrics': self.get_metrics()
        }
        
        # Save results
        self.save_results(results, output_dir)
        
        self.logger.info(f"Filtering complete: {len(valid_tracks)}/{self.total_tracks} tracks passed")
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for this stage.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'total_tracks': self.total_tracks,
            'valid_tracks': len(self.filtered_tracks),
            'filter_rate': len(self.filtered_tracks) / max(1, self.total_tracks),
            'avg_detections_per_track': (
                sum(stats['num_detections'] for stats in self.track_stats.values()) / 
                max(1, len(self.track_stats))
            ),
            'avg_track_duration': (
                sum(stats['duration'] for stats in self.track_stats.values()) / 
                max(1, len(self.track_stats))
            )
        }