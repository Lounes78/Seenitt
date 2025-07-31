"""Stage 6: Final Validation - Validate images using Qwen 2.5VL API."""

import requests
import base64
import time
import json
import re
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shutil

from .base_stage import BaseStage
from ..utils import ensure_dir


class FinalValidationStage(BaseStage):
    """Final validation stage using Qwen 2.5VL API for plant identification."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize final validation stage.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "final_validation")
        
        # API configuration
        validation_config = config.get('validation', {})
        self.api_endpoint = validation_config.get('api_endpoint', '')
        self.timeout = validation_config.get('timeout', 30)
        self.retry_attempts = validation_config.get('retry_attempts', 3)
        self.min_validation_score = validation_config.get('min_validation_score', 40)  # Lower threshold
        self.request_delay = validation_config.get('request_delay', 0.1)  # seconds between requests
        
        # Parallel processing configuration
        self.enable_parallel = validation_config.get('enable_parallel', True)
        self.max_workers = validation_config.get('max_workers', 22)  # Match vLLM capacity
        
        # Validation prompt configuration
        self.validation_prompt = validation_config.get('validation_prompt', self._get_default_prompt())
        self.response_format = validation_config.get('response_format', 'answer: yes/no, score: X')
        
        # Plant identification prompts
        self.identification_prompt = validation_config.get('identification_prompt', self._get_identification_prompt())
        self.enable_plant_identification = validation_config.get('enable_plant_identification', True)
        
        self.processed_images = 0
        self.validated_images = 0
        self.validation_results = {}
        self.api_stats = {'successful_calls': 0, 'failed_calls': 0, 'total_calls': 0}
    
    def _get_default_prompt(self) -> str:
        """Get default validation prompt."""
        return (
            "I am developing a smart glasses app that detects plants and trees in real time. "
            "For each cropped image, please answer: Is this a clear photo of a plant or tree that would be "
            "useful for botanical identification? "
            "REQUIREMENTS: "
            "1. The image should show vegetation (plants, trees, flowers, leaves, branches) "
            "2. The vegetation should be reasonably clear and identifiable "
            "3. REJECT images that show ONLY bare tree trunks without visible leaves, branches, or foliage "
            "4. ACCEPT images even if they show multiple plants or partial plants, as long as vegetation is clearly visible "
            "Answer \"yes\" if the image shows identifiable vegetation (not just bare trunk), \"no\" otherwise. "
            "Give a quality score from 0 (worst) to 100 (best) based on clarity and usefulness for plant identification. "
            "Respond in the exact format: answer: yes/no, score: X "
            "where X is an integer between 0 and 100."
        )
    
    def _get_identification_prompt(self) -> str:
        """Get plant identification prompt."""
        return (
            "Please identify this plant or tree. Provide the following information in JSON format:\\n"
            "{\\n"
            "  \"common_name\": \"Common name of the plant\",\\n"
            "  \"scientific_name\": \"Scientific name (if known)\",\\n"
            "  \"plant_type\": \"tree/shrub/flower/herb/grass/other\",\\n"
            "  \"confidence\": 0-100,\\n"
            "  \"description\": \"Brief description of identifying features\",\\n"
            "  \"season\": \"Best viewing season if applicable\"\\n"
            "}"
        )
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {e}")
            return ""
    
    def _make_api_request(self, prompt: str, image_b64: str) -> Optional[str]:
        """Make API request to validation endpoint.
        
        Args:
            prompt: Prompt for the API
            image_b64: Base64 encoded image
            
        Returns:
            API response text or None if failed
        """
        if not self.api_endpoint:
            self.logger.warning("No API endpoint configured, skipping validation")
            return None
        
        payload = {
            "prompt": prompt,
            "image": image_b64
        }
        
        for attempt in range(self.retry_attempts):
            try:
                self.api_stats['total_calls'] += 1
                
                response = requests.post(
                    self.api_endpoint,
                    headers={'Content-Type': 'application/json'},
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    response_text = response_json.get("response", "").strip()
                    self.api_stats['successful_calls'] += 1
                    return response_text
                else:
                    self.logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API request attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(1)  # Wait before retry
        
        self.api_stats['failed_calls'] += 1
        return None
    
    def _parse_validation_response(self, response_text: str) -> Tuple[Optional[bool], Optional[int]]:
        """Parse validation response from API with improved parsing.
        
        Args:
            response_text: Raw response text from API
            
        Returns:
            Tuple of (answer, score) or (None, None) if parsing failed
        """
        try:
            response_lower = response_text.lower().replace(" ", "")
            
            # Extract answer with multiple patterns
            if "answer:yes" in response_lower or "yes," in response_lower or response_text.lower().startswith("yes"):
                answer = True
            elif "answer:no" in response_lower or "no," in response_lower or response_text.lower().startswith("no"):
                answer = False
            else:
                if re.search(r'\byes\b', response_lower):
                    answer = True
                elif re.search(r'\bno\b', response_lower):
                    answer = False
                else:
                    return None, None
            
            # Extract score with improved pattern matching
            score_match = re.search(r'score:?(\d+)', response_lower)
            if score_match:
                score = int(score_match.group(1))
            else:
                # Look for any number in the response as fallback
                number_match = re.search(r'(\d+)', response_lower)
                if number_match:
                    score = int(number_match.group(1))
                else:
                    score = 50  # Default score
            
            return answer, score
            
        except Exception as e:
            self.logger.warning(f"Failed to parse validation response: '{response_text}' - {e}")
            return None, None
    
    def _parse_identification_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse plant identification response from API.
        
        Args:
            response_text: Raw response text from API
            
        Returns:
            Parsed identification data or None if parsing failed
        """
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                identification_data = json.loads(json_str)
                return identification_data
            else:
                self.logger.warning(f"No JSON found in identification response: {response_text}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to parse identification response: '{response_text}' - {e}")
            return None
    
    def _validate_single_image(self, image_path: str) -> Dict[str, Any]:
        """Validate a single image using the API.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with validation results
        """
        # Encode image
        image_b64 = self._encode_image_to_base64(image_path)
        if not image_b64:
            return {
                'validation_passed': False,
                'validation_score': 0,
                'error': 'Failed to encode image'
            }
        
        # Perform validation
        validation_response = self._make_api_request(self.validation_prompt, image_b64)
        
        if validation_response is None:
            return {
                'validation_passed': False,
                'validation_score': 0,
                'error': 'API request failed'
            }
        
        # Parse validation response
        answer, score = self._parse_validation_response(validation_response)
        
        if answer is None or score is None:
            return {
                'validation_passed': False,
                'validation_score': 0,
                'error': 'Failed to parse validation response',
                'raw_response': validation_response
            }
        
        # Check if validation passed
        validation_passed = answer and score >= self.min_validation_score
        
        result = {
            'validation_passed': validation_passed,
            'validation_answer': answer,
            'validation_score': score,
            'raw_validation_response': validation_response
        }
        
        # Perform plant identification if validation passed and enabled
        if validation_passed and self.enable_plant_identification:
            time.sleep(self.request_delay)  # Delay between requests
            
            identification_response = self._make_api_request(self.identification_prompt, image_b64)
            
            if identification_response:
                identification_data = self._parse_identification_response(identification_response)
                if identification_data:
                    result['plant_identification'] = identification_data
                    result['raw_identification_response'] = identification_response
                else:
                    result['identification_error'] = 'Failed to parse identification response'
                    result['raw_identification_response'] = identification_response
            else:
                result['identification_error'] = 'Identification API request failed'
        
        return result
    
    def _validate_image_parallel(self, crop: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Validate a single image for parallel processing.
        
        Args:
            crop: Crop data dictionary
            
        Returns:
            Tuple of (track_id, validation_result)
        """
        track_id = crop.get('track_id')
        image_path = crop.get('high_quality_filepath', 
                           crop.get('unique_filepath', 
                                   crop.get('selected_filepath', 
                                           crop.get('filepath'))))
        
        if not image_path or not Path(image_path).exists():
            return track_id, {
                'validation_passed': False,
                'validation_score': 0,
                'error': f'Image path not found: {image_path}'
            }
        
        # Use the existing validation method
        return track_id, self._validate_single_image(image_path)
    
    def _process_parallel(self, high_quality_crops: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """Process images using parallel validation.
        
        Args:
            high_quality_crops: List of high quality crop data
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"Starting parallel validation with {self.max_workers} workers")
        
        start_time = time.time()
        validation_results = {}
        validated_crops = []
        plant_identifications = {}
        
        # Create output directory for validated images
        validated_dir = Path(output_dir) / "validated_crops"
        validated_dir.mkdir(exist_ok=True)
        
        # Progress tracking
        completed = 0
        lock = threading.Lock()
        
        def update_progress():
            nonlocal completed
            with lock:
                completed += 1
                if completed % 10 == 0 or completed == len(high_quality_crops):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(high_quality_crops) - completed) / rate if rate > 0 else 0
                    self.logger.info(f"Progress: {completed}/{len(high_quality_crops)} "
                                   f"({completed/len(high_quality_crops)*100:.1f}%) - "
                                   f"Rate: {rate:.1f}/s - ETA: {remaining:.0f}s")
        
        # Process with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_crop = {executor.submit(self._validate_image_parallel, crop): crop 
                            for crop in high_quality_crops}
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_crop):
                crop = future_to_crop[future]
                try:
                    track_id, result = future.result()
                    validation_results[track_id] = result
                    
                    # If validation passed, copy to output directory
                    if result.get('validation_passed', False):
                        image_path = crop.get('high_quality_filepath', 
                                           crop.get('unique_filepath', 
                                                   crop.get('selected_filepath', 
                                                           crop.get('filepath'))))
                        
                        new_filename = f"validated_track_{track_id}.jpg"
                        new_path = validated_dir / new_filename
                        
                        try:
                            shutil.copy2(image_path, new_path)
                            
                            # Update crop info
                            validated_crop = crop.copy()
                            validated_crop['validated_filepath'] = str(new_path)
                            validated_crop['validated_filename'] = new_filename
                            validated_crop['validation_result'] = result
                            validated_crops.append(validated_crop)
                            
                            # Store plant identification if available
                            if 'plant_identification' in result:
                                plant_identifications[track_id] = result['plant_identification']
                            
                            self.validated_images += 1
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to copy validated crop {image_path}: {e}")
                    
                    update_progress()
                    
                except Exception as e:
                    self.logger.error(f"Error processing crop {crop.get('track_id')}: {e}")
                    update_progress()
        
        processing_time = time.time() - start_time
        self.logger.info(f"Parallel validation completed in {processing_time:.1f} seconds "
                        f"(rate: {len(high_quality_crops)/processing_time:.1f} images/s)")
        
        return {
            'validated_crops': validated_crops,
            'validation_results': validation_results,
            'plant_identifications': plant_identifications,
            'validated_directory': str(validated_dir),
            'processing_metrics': {
                'processing_time': processing_time,
                'processing_rate': len(high_quality_crops)/processing_time,
                'parallel_workers': self.max_workers
            }
        }
    
    def process(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Main processing method for final validation.
        
        Args:
            input_data: Results from quality assessment stage
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Starting final validation with Qwen 2.5VL API")
        
        high_quality_crops = input_data['high_quality_crops']
        self.processed_images = len(high_quality_crops)
        
        if self.processed_images == 0:
            self.logger.warning("No high-quality crops to validate")
            return {
                'validated_crops': [],
                'validation_results': {},
                'plant_identifications': {},
                'validated_directory': '',
                'stage_metrics': self.get_metrics()
            }
        
        # Choose processing mode
        if self.enable_parallel and self.processed_images > 1:
            self.logger.info(f"Using parallel processing with {self.max_workers} workers")
            parallel_results = self._process_parallel(high_quality_crops, output_dir)
            
            # Extract results from parallel processing
            validated_crops = parallel_results['validated_crops']
            validation_results = parallel_results['validation_results']
            plant_identifications = parallel_results['plant_identifications']
            validated_dir = Path(parallel_results['validated_directory'])
            
        else:
            # Fall back to sequential processing
            self.logger.info("Using sequential processing")
            
            # Create output directory for validated images
            validated_dir = Path(output_dir) / "validated_crops"
            ensure_dir(validated_dir)
            
            validated_crops = []
            plant_identifications = {}
            validation_results = {}
            
            # Process each image sequentially
            for i, crop in enumerate(high_quality_crops):
                image_path = crop.get('high_quality_filepath', 
                                   crop.get('unique_filepath', 
                                           crop.get('selected_filepath', 
                                                   crop.get('filepath'))))
                
                if not image_path or not Path(image_path).exists():
                    self.logger.warning(f"Image path not found: {image_path}")
                    continue
                
                track_id = crop.get('track_id')
                self.logger.info(f"Validating image {i+1}/{self.processed_images} (track {track_id})")
                
                # Validate image
                validation_result = self._validate_single_image(image_path)
                validation_results[track_id] = validation_result
                
                # If validation passed, copy to output directory
                if validation_result.get('validation_passed', False):
                    new_filename = f"validated_track_{track_id}.jpg"
                    new_path = validated_dir / new_filename
                    
                    try:
                        shutil.copy2(image_path, new_path)
                        
                        # Update crop info
                        validated_crop = crop.copy()
                        validated_crop['validated_filepath'] = str(new_path)
                        validated_crop['validated_filename'] = new_filename
                        validated_crop['validation_result'] = validation_result
                        validated_crops.append(validated_crop)
                        
                        # Store plant identification if available
                        if 'plant_identification' in validation_result:
                            plant_identifications[track_id] = validation_result['plant_identification']
                        
                        self.validated_images += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to copy validated crop {image_path}: {e}")
                else:
                    # Log validation failure reason
                    error = validation_result.get('error', 'Unknown error')
                    score = validation_result.get('validation_score', 0)
                    self.logger.debug(f"Track {track_id} validation failed: {error} (score: {score})")
                
                # Add delay between requests to avoid rate limiting
                if i < self.processed_images - 1:  # Don't delay after last image
                    time.sleep(self.request_delay)
        
        # Generate summary
        summary_data = self._generate_summary(validated_crops, plant_identifications)
        
        # Compile results
        results = {
            'validated_crops': validated_crops,
            'validation_results': validation_results,
            'plant_identifications': plant_identifications,
            'validated_directory': str(validated_dir),
            'summary': summary_data,
            'validation_config': {
                'api_endpoint': self.api_endpoint,
                'min_validation_score': self.min_validation_score,
                'enable_plant_identification': self.enable_plant_identification
            },
            'api_statistics': self.api_stats,
            'stage_metrics': self.get_metrics()
        }
        
        # Save results
        self.save_results(results, output_dir)
        
        # Save summary as separate file
        summary_path = Path(output_dir) / "plant_summary.json"
        with open(summary_path, 'w') as f:
            # Ensure summary is JSON serializable
            serializable_summary = self._make_json_serializable(summary_data)
            json.dump(serializable_summary, f, indent=2)
        
        self.logger.info(f"Final validation complete: {self.validated_images} validated images "
                        f"from {self.processed_images} input images "
                        f"(validation rate: {self.validated_images/max(1, self.processed_images)*100:.1f}%)")
        
        return results
    
    def _generate_summary(self, validated_crops: List[Dict[str, Any]], 
                         plant_identifications: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of validated plants for user consumption.
        
        Args:
            validated_crops: List of validated crop data
            plant_identifications: Plant identification data by track ID
            
        Returns:
            Summary data for end user
        """
        plants_found = []
        
        for crop in validated_crops:
            track_id = crop.get('track_id')
            validation_result = crop.get('validation_result', {})
            
            plant_info = {
                'track_id': track_id,
                'image_path': crop.get('validated_filepath'),
                'validation_score': validation_result.get('validation_score', 0),
                'frame_captured': crop.get('frame', 0)
            }
            
            # Add identification info if available
            if track_id in plant_identifications:
                identification = plant_identifications[track_id]
                plant_info.update({
                    'common_name': identification.get('common_name', 'Unknown'),
                    'scientific_name': identification.get('scientific_name', 'Unknown'),
                    'plant_type': identification.get('plant_type', 'Unknown'),
                    'identification_confidence': identification.get('confidence', 0),
                    'description': identification.get('description', ''),
                    'season': identification.get('season', '')
                })
            else:
                plant_info.update({
                    'common_name': 'Plant/Tree',
                    'scientific_name': 'Unknown',
                    'plant_type': 'Unknown',
                    'identification_confidence': 0,
                    'description': 'High-quality plant image validated',
                    'season': ''
                })
            
            plants_found.append(plant_info)
        
        # Sort by validation score (highest first)
        plants_found.sort(key=lambda x: x['validation_score'], reverse=True)
        
        # Generate statistics
        plant_types = [p.get('plant_type', 'Unknown') for p in plants_found if p.get('plant_type') != 'Unknown']
        type_counts = {}
        for plant_type in plant_types:
            type_counts[plant_type] = type_counts.get(plant_type, 0) + 1
        
        summary = {
            'total_plants_found': len(plants_found),
            'plants_with_identification': len(plant_identifications),
            'plant_type_distribution': type_counts,
            'average_validation_score': (
                sum(p['validation_score'] for p in plants_found) / max(1, len(plants_found))
            ),
            'plants': plants_found,
            'generation_timestamp': time.time()
        }
        
        return summary
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # NumPy scalar
            return obj.item()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        else:
            return obj
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for this stage.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'input_images': self.processed_images,
            'validated_images': self.validated_images,
            'validation_pass_rate': self.validated_images / max(1, self.processed_images),
            'api_success_rate': self.api_stats['successful_calls'] / max(1, self.api_stats['total_calls']),
            'total_api_calls': self.api_stats['total_calls'],
            'successful_api_calls': self.api_stats['successful_calls'],
            'failed_api_calls': self.api_stats['failed_calls'],
            'min_validation_score': self.min_validation_score
        }