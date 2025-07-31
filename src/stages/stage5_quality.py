"""Stage 5: Quality Assessment - Filter images based on quality metrics."""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import shutil

from torchmetrics.multimodal import CLIPImageQualityAssessment
from .base_stage import BaseStage
from ..utils import ensure_dir


class QualityAssessmentStage(BaseStage):
    """Assess and filter images based on quality metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize quality assessment stage.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "quality_assessment")
        
        # Quality configuration
        quality_config = config.get('quality', {})
        self.min_quality_score = quality_config.get('min_quality_score', 0.3)
        self.clip_model_name = quality_config.get('clip_model', 'openai/clip-vit-base-patch32')
        self.image_size = quality_config.get('image_size', [224, 224])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Additional quality criteria
        self.min_resolution = quality_config.get('min_resolution', [100, 100])  # [width, height]
        self.max_blur_variance = quality_config.get('max_blur_variance', 50)  # Lower = more blurry
        self.min_brightness = quality_config.get('min_brightness', 20)
        self.max_brightness = quality_config.get('max_brightness', 235)
        self.min_contrast_std = quality_config.get('min_contrast_std', 15)
        
        # Enable/disable specific quality checks
        self.enable_clip_quality = quality_config.get('enable_clip_quality', True)
        self.enable_blur_detection = quality_config.get('enable_blur_detection', True)
        self.enable_brightness_check = quality_config.get('enable_brightness_check', True)
        self.enable_contrast_check = quality_config.get('enable_contrast_check', True)
        self.enable_resolution_check = quality_config.get('enable_resolution_check', True)
        
        # Initialize CLIP quality model
        self.quality_metric = None
        if self.enable_clip_quality:
            self._load_quality_model()
        
        self.processed_images = 0
        self.high_quality_images = 0
        self.quality_stats = {}
    
    def _load_quality_model(self) -> None:
        """Load CLIP quality assessment model."""
        try:
            self.quality_metric = CLIPImageQualityAssessment().to(self.device)
            self.logger.info(f"CLIP quality model loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP quality model: {e}")
            self.enable_clip_quality = False
    
    def _calculate_clip_quality(self, image: np.ndarray) -> float:
        """Calculate CLIP-based quality score.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Quality score between 0 and 1
        """
        if not self.enable_clip_quality or self.quality_metric is None:
            return 1.0
        
        try:
            # Resize image to model input size
            img_resized = cv2.resize(image, tuple(self.image_size))
            
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
            self.logger.warning(f"Failed to calculate CLIP quality: {e}")
            return 1.0
    
    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            Blur score (higher = less blurry)
        """
        if not self.enable_blur_detection:
            return float(self.max_blur_variance + 1)
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            return float(variance)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate blur score: {e}")
            return float(self.max_blur_variance + 1)
    
    def _calculate_brightness_stats(self, image: np.ndarray) -> Tuple[float, bool]:
        """Calculate brightness statistics.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            Tuple of (average brightness, is_valid)
        """
        if not self.enable_brightness_check:
            return 128.0, True
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate average brightness
            avg_brightness = np.mean(gray)
            
            # Check if within acceptable range
            is_valid = self.min_brightness <= avg_brightness <= self.max_brightness
            
            return float(avg_brightness), is_valid
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate brightness: {e}")
            return 128.0, True
    
    def _calculate_contrast_stats(self, image: np.ndarray) -> Tuple[float, bool]:
        """Calculate contrast statistics.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            Tuple of (contrast standard deviation, is_valid)
        """
        if not self.enable_contrast_check:
            return float(self.min_contrast_std + 1), True
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate standard deviation as contrast measure
            contrast_std = np.std(gray)
            
            # Check if above minimum threshold
            is_valid = contrast_std >= self.min_contrast_std
            
            return float(contrast_std), is_valid
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate contrast: {e}")
            return float(self.min_contrast_std + 1), True
    
    def _check_resolution(self, image: np.ndarray) -> Tuple[Tuple[int, int], bool]:
        """Check image resolution.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of ((width, height), is_valid)
        """
        if not self.enable_resolution_check:
            return (image.shape[1], image.shape[0]), True
        
        height, width = image.shape[:2]
        min_width, min_height = self.min_resolution
        
        is_valid = width >= min_width and height >= min_height
        
        return (width, height), is_valid
    
    def _assess_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Assess quality of a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with quality assessment results
        """
        try:
            # Load image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                return {
                    'valid': False,
                    'error': 'Failed to load image',
                    'clip_quality': 0.0,
                    'blur_score': 0.0,
                    'brightness': 0.0,
                    'contrast_std': 0.0,
                    'resolution': (0, 0)
                }
            
            # Convert to RGB for CLIP
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Calculate various quality metrics
            clip_quality = self._calculate_clip_quality(image_rgb)
            blur_score = self._calculate_blur_score(image_bgr)
            brightness, brightness_valid = self._calculate_brightness_stats(image_bgr)
            contrast_std, contrast_valid = self._calculate_contrast_stats(image_bgr)
            resolution, resolution_valid = self._check_resolution(image_bgr)
            
            # Determine overall validity
            clip_valid = clip_quality >= self.min_quality_score if self.enable_clip_quality else True
            blur_valid = blur_score >= self.max_blur_variance if self.enable_blur_detection else True
            
            overall_valid = all([
                clip_valid,
                blur_valid,
                brightness_valid,
                contrast_valid,
                resolution_valid
            ])
            
            return {
                'valid': overall_valid,
                'clip_quality': clip_quality,
                'clip_valid': clip_valid,
                'blur_score': blur_score,
                'blur_valid': blur_valid,
                'brightness': brightness,
                'brightness_valid': brightness_valid,
                'contrast_std': contrast_std,
                'contrast_valid': contrast_valid,
                'resolution': resolution,
                'resolution_valid': resolution_valid,
                'file_size': Path(image_path).stat().st_size if Path(image_path).exists() else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Error assessing quality for {image_path}: {e}")
            return {
                'valid': False,
                'error': str(e),
                'clip_quality': 0.0,
                'blur_score': 0.0,
                'brightness': 0.0,
                'contrast_std': 0.0,
                'resolution': (0, 0)
            }
    
    def process(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Main processing method for quality assessment.
        
        Args:
            input_data: Results from similarity detection stage
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing quality assessment results
        """
        self.logger.info("Starting quality assessment")
        
        unique_crops = input_data['unique_crops']
        self.processed_images = len(unique_crops)
        
        if self.processed_images == 0:
            self.logger.warning("No unique crops to assess")
            return {
                'high_quality_crops': [],
                'quality_assessments': {},
                'high_quality_directory': '',
                'stage_metrics': self.get_metrics()
            }
        
        # Create output directory for high-quality images
        high_quality_dir = Path(output_dir) / "high_quality_crops"
        ensure_dir(high_quality_dir)
        
        high_quality_crops = []
        quality_assessments = {}
        
        # Assess each image
        for crop in unique_crops:
            image_path = crop.get('unique_filepath', 
                               crop.get('selected_filepath', 
                                       crop.get('filtered_filepath', 
                                               crop.get('filepath'))))
            
            if not image_path or not Path(image_path).exists():
                self.logger.warning(f"Image path not found: {image_path}")
                continue
            
            # Assess quality
            quality_result = self._assess_image_quality(image_path)
            track_id = crop.get('track_id')
            quality_assessments[track_id] = quality_result
            
            # If high quality, copy to output directory
            if quality_result['valid']:
                new_filename = f"high_quality_track_{track_id}.jpg"
                new_path = high_quality_dir / new_filename
                
                try:
                    shutil.copy2(image_path, new_path)
                    
                    # Update crop info
                    high_quality_crop = crop.copy()
                    high_quality_crop['high_quality_filepath'] = str(new_path)
                    high_quality_crop['high_quality_filename'] = new_filename
                    high_quality_crop['quality_assessment'] = quality_result
                    high_quality_crops.append(high_quality_crop)
                    
                    self.high_quality_images += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to copy high-quality crop {image_path}: {e}")
            else:
                # Log why the image was rejected
                reasons = []
                if not quality_result.get('clip_valid', True):
                    reasons.append(f"low CLIP quality ({quality_result['clip_quality']:.3f})")
                if not quality_result.get('blur_valid', True):
                    reasons.append(f"too blurry ({quality_result['blur_score']:.1f})")
                if not quality_result.get('brightness_valid', True):
                    reasons.append(f"poor brightness ({quality_result['brightness']:.1f})")
                if not quality_result.get('contrast_valid', True):
                    reasons.append(f"low contrast ({quality_result['contrast_std']:.1f})")
                if not quality_result.get('resolution_valid', True):
                    reasons.append(f"low resolution {quality_result['resolution']}")
                
                self.logger.debug(f"Track {track_id} rejected: {', '.join(reasons)}")
        
        # Calculate summary statistics
        if quality_assessments:
            self.quality_stats = {
                'avg_clip_quality': np.mean([qa['clip_quality'] for qa in quality_assessments.values()]),
                'avg_blur_score': np.mean([qa['blur_score'] for qa in quality_assessments.values()]),
                'avg_brightness': np.mean([qa['brightness'] for qa in quality_assessments.values()]),
                'avg_contrast': np.mean([qa['contrast_std'] for qa in quality_assessments.values()]),
                'clip_pass_rate': sum(1 for qa in quality_assessments.values() if qa.get('clip_valid', True)) / len(quality_assessments),
                'blur_pass_rate': sum(1 for qa in quality_assessments.values() if qa.get('blur_valid', True)) / len(quality_assessments),
                'brightness_pass_rate': sum(1 for qa in quality_assessments.values() if qa.get('brightness_valid', True)) / len(quality_assessments),
                'contrast_pass_rate': sum(1 for qa in quality_assessments.values() if qa.get('contrast_valid', True)) / len(quality_assessments),
                'resolution_pass_rate': sum(1 for qa in quality_assessments.values() if qa.get('resolution_valid', True)) / len(quality_assessments)
            }
        
        # Compile results
        results = {
            'high_quality_crops': high_quality_crops,
            'quality_assessments': quality_assessments,
            'quality_statistics': self.quality_stats,
            'high_quality_directory': str(high_quality_dir),
            'quality_criteria': {
                'min_quality_score': self.min_quality_score,
                'min_resolution': self.min_resolution,
                'max_blur_variance': self.max_blur_variance,
                'min_brightness': self.min_brightness,
                'max_brightness': self.max_brightness,
                'min_contrast_std': self.min_contrast_std
            },
            'enabled_checks': {
                'clip_quality': self.enable_clip_quality,
                'blur_detection': self.enable_blur_detection,
                'brightness_check': self.enable_brightness_check,
                'contrast_check': self.enable_contrast_check,
                'resolution_check': self.enable_resolution_check
            },
            'stage_metrics': self.get_metrics()
        }
        
        # Save results
        self.save_results(results, output_dir)
        
        self.logger.info(f"Quality assessment complete: {self.high_quality_images} high-quality images "
                        f"from {self.processed_images} input images "
                        f"(pass rate: {self.high_quality_images/max(1, self.processed_images)*100:.1f}%)")
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for this stage.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'input_images': self.processed_images,
            'high_quality_images': self.high_quality_images,
            'quality_pass_rate': self.high_quality_images / max(1, self.processed_images),
            'quality_criteria_used': {
                'clip_quality': self.enable_clip_quality,
                'blur_detection': self.enable_blur_detection,
                'brightness_check': self.enable_brightness_check,
                'contrast_check': self.enable_contrast_check,
                'resolution_check': self.enable_resolution_check
            }
        }
        
        # Add quality statistics if available
        if self.quality_stats:
            metrics.update(self.quality_stats)
        
        return metrics