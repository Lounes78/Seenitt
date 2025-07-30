"""Stage 1: Object Detection & Tracking using YOLO."""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from ultralytics import YOLO
import torch

from .base_stage import BaseStage
from ..utils import ensure_dir


class YOLODetectionStage(BaseStage):
    """YOLO-based object detection and tracking stage."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YOLO detection stage.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "detection_tracking")
        
        # YOLO configuration
        yolo_config = config.get('yolo', {})
        self.model_path = yolo_config.get('model_path', 'yolov8x-oiv7.pt')
        self.confidence_threshold = yolo_config.get('confidence_threshold', 0.3)
        self.device = yolo_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model
        self.model = None
        self._load_model()
        
        # Filtering configuration
        filtering_config = config.get('filtering', {})
        self.plant_classes = filtering_config.get('plant_classes', [])
        self.min_bbox_area = filtering_config.get('min_bbox_area', 1000)
        self.max_bbox_area = filtering_config.get('max_bbox_area', 500000)
        
        self.processed_frames = 0
        self.total_detections = 0
        self.filtered_detections = 0
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _calculate_bbox_area(self, bbox: List[float]) -> float:
        """Calculate bounding box area.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Area of the bounding box
        """
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def _filter_detection(self, obj: Dict[str, Any]) -> bool:
        """Filter detection based on class and bbox criteria.
        
        Args:
            obj: Detection object
            
        Returns:
            True if detection should be kept, False otherwise
        """
        # Filter by class if plant_classes is specified
        if self.plant_classes and obj['class_id'] not in self.plant_classes:
            return False
        
        # Filter by bounding box area
        bbox_area = self._calculate_bbox_area(obj['bbox'])
        if bbox_area < self.min_bbox_area or bbox_area > self.max_bbox_area:
            return False
        
        return True
    
    def process_video_stream(self, video_source: str, output_dir: str, 
                           show_display: bool = False) -> Dict[str, Any]:
        """Process video stream for object detection and tracking.
        
        Args:
            video_source: Path to video file or camera index
            output_dir: Directory to save results
            show_display: Whether to show real-time display
            
        Returns:
            Dictionary containing detection results
        """
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        self.logger.info(f"Processing video source: {video_source}")
        
        # Run YOLO tracking
        results = self.model.track(
            source=video_source,
            show=show_display,
            stream=True,
            conf=self.confidence_threshold,
            device=self.device
        )
        
        video_data = []
        
        for frame_idx, r in enumerate(results):
            self.processed_frames += 1
            
            frame_data = {
                'frame': frame_idx,
                'timestamp': frame_idx / 30.0,  # Assume 30 FPS, adjust as needed
                'objects': []
            }
            
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy().tolist()
                confidences = r.boxes.conf.cpu().numpy().tolist()
                class_ids = r.boxes.cls.cpu().numpy().tolist()
                track_ids = r.boxes.id.cpu().numpy().tolist() if r.boxes.id is not None else []
                
                for i in range(len(boxes)):
                    self.total_detections += 1
                    
                    obj = {
                        'track_id': int(track_ids[i]) if track_ids and i < len(track_ids) else None,
                        'class_id': int(class_ids[i]),
                        'bbox': boxes[i],
                        'confidence': confidences[i],
                        'bbox_area': self._calculate_bbox_area(boxes[i])
                    }
                    
                    # Apply filtering
                    if self._filter_detection(obj):
                        frame_data['objects'].append(obj)
                        self.filtered_detections += 1
            
            video_data.append(frame_data)
            
            # Log progress every 100 frames
            if frame_idx % 100 == 0:
                self.logger.info(f"Processed {frame_idx} frames, "
                               f"Total detections: {self.total_detections}, "
                               f"Filtered: {self.filtered_detections}")
        
        # Save results
        results_data = {
            'video_source': str(video_source),
            'total_frames': len(video_data),
            'total_detections': self.total_detections,
            'filtered_detections': self.filtered_detections,
            'model_info': {
                'model_path': self.model_path,
                'confidence_threshold': self.confidence_threshold,
                'device': self.device
            },
            'filtering_criteria': {
                'plant_classes': self.plant_classes,
                'min_bbox_area': self.min_bbox_area,
                'max_bbox_area': self.max_bbox_area
            },
            'frames': video_data
        }
        
        return results_data
    
    def process_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame for detection.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Detection results for the frame
        """
        results = self.model.track(
            source=frame,
            stream=False,
            conf=self.confidence_threshold,
            device=self.device
        )
        
        frame_data = {'objects': []}
        
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy().tolist()
                confidences = r.boxes.conf.cpu().numpy().tolist()
                class_ids = r.boxes.cls.cpu().numpy().tolist()
                track_ids = r.boxes.id.cpu().numpy().tolist() if r.boxes.id is not None else []
                
                for i in range(len(boxes)):
                    obj = {
                        'track_id': int(track_ids[i]) if track_ids and i < len(track_ids) else None,
                        'class_id': int(class_ids[i]),
                        'bbox': boxes[i],
                        'confidence': confidences[i],
                        'bbox_area': self._calculate_bbox_area(boxes[i])
                    }
                    
                    if self._filter_detection(obj):
                        frame_data['objects'].append(obj)
        
        return frame_data
    
    def crop_detections(self, video_source: str, detection_results: Dict[str, Any], 
                       output_dir: str) -> Dict[str, Any]:
        """Crop detected objects from video frames.
        
        Args:
            video_source: Path to video file
            detection_results: Results from detection stage
            output_dir: Directory to save cropped images
            
        Returns:
            Dictionary with cropping results
        """
        crop_dir = Path(output_dir) / "crops"
        ensure_dir(crop_dir)
        
        cap = cv2.VideoCapture(video_source)
        cropped_images = []
        total_crops = 0
        
        for frame_data in detection_results['frames']:
            frame_number = frame_data['frame']
            objects = frame_data['objects']
            
            # Set video to correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            for idx, obj in enumerate(objects):
                track_id = obj['track_id']
                class_id = obj['class_id']
                bbox = obj['bbox']
                confidence = obj['confidence']
                
                # Crop image
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:  # Valid crop
                    cropped_image = frame[y1:y2, x1:x2]
                    
                    # Generate filename
                    filename = f"class_{class_id}_track_{track_id}_frame_{frame_number}_bbox_{idx}.jpg"
                    filepath = crop_dir / filename
                    
                    # Save cropped image
                    cv2.imwrite(str(filepath), cropped_image)
                    
                    cropped_images.append({
                        'filename': filename,
                        'filepath': str(filepath),
                        'track_id': track_id,
                        'class_id': class_id,
                        'frame': frame_number,
                        'bbox': bbox,
                        'confidence': confidence,
                        'crop_size': [x2-x1, y2-y1]
                    })
                    
                    total_crops += 1
        
        cap.release()
        
        crop_results = {
            'total_crops': total_crops,
            'crop_directory': str(crop_dir),
            'cropped_images': cropped_images
        }
        
        self.logger.info(f"Cropped {total_crops} images to {crop_dir}")
        return crop_results
    
    def process(self, input_data: str, output_dir: str) -> Dict[str, Any]:
        """Main processing method for the detection stage.
        
        Args:
            input_data: Path to video file
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all stage results
        """
        self.logger.info(f"Starting detection and tracking for: {input_data}")
        
        # Process video for detection and tracking
        detection_results = self.process_video_stream(input_data, output_dir)
        
        # Crop detected objects
        crop_results = self.crop_detections(input_data, detection_results, output_dir)
        
        # Combine results
        combined_results = {
            'detection_results': detection_results,
            'crop_results': crop_results,
            'stage_metrics': self.get_metrics()
        }
        
        # Save results
        self.save_results(combined_results, output_dir)
        
        return combined_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for this stage.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'processed_frames': self.processed_frames,
            'total_detections': self.total_detections,
            'filtered_detections': self.filtered_detections,
            'filter_rate': self.filtered_detections / max(1, self.total_detections),
            'avg_detections_per_frame': self.total_detections / max(1, self.processed_frames)
        }