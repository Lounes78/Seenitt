"""Stage 6: Final Validation - Validate images using Qwen 2.5VL API."""

import requests
import base64
import time
import json
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
        self.min_validation_score = validation_config.get('min_validation_score', 70)
        self.request_delay = validation_config.get('request_delay', 0.3)  # seconds between requests
        
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
            "For each cropped image, please answer: Is this a high-quality, well-framed, clear, "
            "and centered close-up photo of ONE SINGLE, COMPLETE plant or tree suitable for display in a botanical identification app? "
            "CRITICAL REQUIREMENTS: "
            "1. There must be ONLY ONE individual plant or tree visible, not multiple plants/trees clustered together "
            "2. The ENTIRE plant or tree must be visible and complete - NOT cropped or cut off at edges "
            "3. No messy, disorganized, or cluttered vegetation - only one clean, isolated specimen "
            "4. The plant/tree should be the clear main subject, well-centered and properly framed "
            "Only answer \"yes\" or \"no\", and give a quality score from 0 (worst) to 100 (best) based on clarity, "
            "framing, lighting, completeness, and suitability for botanical identification. Respond in the exact format: "
            "answer: yes/no, score: X "
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
        """Parse validation response from API.
        
        Args:
            response_text: Raw response text from API
            
        Returns:
            Tuple of (answer, score) or (None, None) if parsing failed
        """
        try:
            # Expected format: answer: yes/no, score: X
            parts = response_text.lower().replace(" ", "").split(",")\n            answer_part = parts[0].split(":")[1]\n            score_part = parts[1].split(":")[1]\n            \n            answer = answer_part == "yes"\n            score = int(score_part)\n            \n            return answer, score\n            \n        except Exception as e:\n            self.logger.warning(f"Failed to parse validation response: '{response_text}' - {e}")\n            return None, None\n    \n    def _parse_identification_response(self, response_text: str) -> Optional[Dict[str, Any]]:\n        """Parse plant identification response from API.\n        \n        Args:\n            response_text: Raw response text from API\n            \n        Returns:\n            Parsed identification data or None if parsing failed\n        """\n        try:\n            # Try to extract JSON from response\n            import re\n            json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)\n            if json_match:\n                json_str = json_match.group(0)\n                identification_data = json.loads(json_str)\n                return identification_data\n            else:\n                self.logger.warning(f"No JSON found in identification response: {response_text}")\n                return None\n                \n        except Exception as e:\n            self.logger.warning(f"Failed to parse identification response: '{response_text}' - {e}")\n            return None\n    \n    def _validate_single_image(self, image_path: str) -> Dict[str, Any]:\n        """Validate a single image using the API.\n        \n        Args:\n            image_path: Path to image file\n            \n        Returns:\n            Dictionary with validation results\n        """\n        # Encode image\n        image_b64 = self._encode_image_to_base64(image_path)\n        if not image_b64:\n            return {\n                'validation_passed': False,\n                'validation_score': 0,\n                'error': 'Failed to encode image'\n            }\n        \n        # Perform validation\n        validation_response = self._make_api_request(self.validation_prompt, image_b64)\n        \n        if validation_response is None:\n            return {\n                'validation_passed': False,\n                'validation_score': 0,\n                'error': 'API request failed'\n            }\n        \n        # Parse validation response\n        answer, score = self._parse_validation_response(validation_response)\n        \n        if answer is None or score is None:\n            return {\n                'validation_passed': False,\n                'validation_score': 0,\n                'error': 'Failed to parse validation response',\n                'raw_response': validation_response\n            }\n        \n        # Check if validation passed\n        validation_passed = answer and score >= self.min_validation_score\n        \n        result = {\n            'validation_passed': validation_passed,\n            'validation_answer': answer,\n            'validation_score': score,\n            'raw_validation_response': validation_response\n        }\n        \n        # Perform plant identification if validation passed and enabled\n        if validation_passed and self.enable_plant_identification:\n            time.sleep(self.request_delay)  # Delay between requests\n            \n            identification_response = self._make_api_request(self.identification_prompt, image_b64)\n            \n            if identification_response:\n                identification_data = self._parse_identification_response(identification_response)\n                if identification_data:\n                    result['plant_identification'] = identification_data\n                    result['raw_identification_response'] = identification_response\n                else:\n                    result['identification_error'] = 'Failed to parse identification response'\n                    result['raw_identification_response'] = identification_response\n            else:\n                result['identification_error'] = 'Identification API request failed'\n        \n        return result\n    \n    def process(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:\n        """Main processing method for final validation.\n        \n        Args:\n            input_data: Results from quality assessment stage\n            output_dir: Directory to save results\n            \n        Returns:\n            Dictionary containing validation results\n        """\n        self.logger.info("Starting final validation with Qwen 2.5VL API")\n        \n        high_quality_crops = input_data['high_quality_crops']\n        self.processed_images = len(high_quality_crops)\n        \n        if self.processed_images == 0:\n            self.logger.warning("No high-quality crops to validate")\n            return {\n                'validated_crops': [],\n                'validation_results': {},\n                'plant_identifications': {},\n                'validated_directory': '',\n                'stage_metrics': self.get_metrics()\n            }\n        \n        # Create output directory for validated images\n        validated_dir = Path(output_dir) / "validated_crops"\n        ensure_dir(validated_dir)\n        \n        validated_crops = []\n        plant_identifications = {}\n        \n        # Process each image\n        for i, crop in enumerate(high_quality_crops):\n            image_path = crop.get('high_quality_filepath', \n                               crop.get('unique_filepath', \n                                       crop.get('selected_filepath', \n                                               crop.get('filepath'))))\n            \n            if not image_path or not Path(image_path).exists():\n                self.logger.warning(f"Image path not found: {image_path}")\n                continue\n            \n            track_id = crop.get('track_id')\n            self.logger.info(f"Validating image {i+1}/{self.processed_images} (track {track_id})")\n            \n            # Validate image\n            validation_result = self._validate_single_image(image_path)\n            self.validation_results[track_id] = validation_result\n            \n            # If validation passed, copy to output directory\n            if validation_result.get('validation_passed', False):\n                new_filename = f"validated_track_{track_id}.jpg"\n                new_path = validated_dir / new_filename\n                \n                try:\n                    shutil.copy2(image_path, new_path)\n                    \n                    # Update crop info\n                    validated_crop = crop.copy()\n                    validated_crop['validated_filepath'] = str(new_path)\n                    validated_crop['validated_filename'] = new_filename\n                    validated_crop['validation_result'] = validation_result\n                    validated_crops.append(validated_crop)\n                    \n                    # Store plant identification if available\n                    if 'plant_identification' in validation_result:\n                        plant_identifications[track_id] = validation_result['plant_identification']\n                    \n                    self.validated_images += 1\n                    \n                except Exception as e:\n                    self.logger.warning(f"Failed to copy validated crop {image_path}: {e}")\n            else:\n                # Log validation failure reason\n                error = validation_result.get('error', 'Unknown error')\n                score = validation_result.get('validation_score', 0)\n                self.logger.debug(f"Track {track_id} validation failed: {error} (score: {score})")\n            \n            # Add delay between requests to avoid rate limiting\n            if i < self.processed_images - 1:  # Don't delay after last image\n                time.sleep(self.request_delay)\n        \n        # Generate summary\n        summary_data = self._generate_summary(validated_crops, plant_identifications)\n        \n        # Compile results\n        results = {\n            'validated_crops': validated_crops,\n            'validation_results': self.validation_results,\n            'plant_identifications': plant_identifications,\n            'validated_directory': str(validated_dir),\n            'summary': summary_data,\n            'validation_config': {\n                'api_endpoint': self.api_endpoint,\n                'min_validation_score': self.min_validation_score,\n                'enable_plant_identification': self.enable_plant_identification\n            },\n            'api_statistics': self.api_stats,\n            'stage_metrics': self.get_metrics()\n        }\n        \n        # Save results\n        self.save_results(results, output_dir)\n        \n        # Save summary as separate file\n        summary_path = Path(output_dir) / "plant_summary.json"\n        with open(summary_path, 'w') as f:\n            json.dump(summary_data, f, indent=2)\n        \n        self.logger.info(f"Final validation complete: {self.validated_images} validated images ")\n                        f"from {self.processed_images} input images ")\n                        f"(validation rate: {self.validated_images/max(1, self.processed_images)*100:.1f}%)")\n        \n        return results\n    \n    def _generate_summary(self, validated_crops: List[Dict[str, Any]], \n                         plant_identifications: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:\n        """Generate summary of validated plants for user consumption.\n        \n        Args:\n            validated_crops: List of validated crop data\n            plant_identifications: Plant identification data by track ID\n            \n        Returns:\n            Summary data for end user\n        """\n        plants_found = []\n        \n        for crop in validated_crops:\n            track_id = crop.get('track_id')\n            validation_result = crop.get('validation_result', {})\n            \n            plant_info = {\n                'track_id': track_id,\n                'image_path': crop.get('validated_filepath'),\n                'validation_score': validation_result.get('validation_score', 0),\n                'frame_captured': crop.get('frame', 0)\n            }\n            \n            # Add identification info if available\n            if track_id in plant_identifications:\n                identification = plant_identifications[track_id]\n                plant_info.update({\n                    'common_name': identification.get('common_name', 'Unknown'),\n                    'scientific_name': identification.get('scientific_name', 'Unknown'),\n                    'plant_type': identification.get('plant_type', 'Unknown'),\n                    'identification_confidence': identification.get('confidence', 0),\n                    'description': identification.get('description', ''),\n                    'season': identification.get('season', '')\n                })\n            else:\n                plant_info.update({\n                    'common_name': 'Plant/Tree',\n                    'scientific_name': 'Unknown',\n                    'plant_type': 'Unknown',\n                    'identification_confidence': 0,\n                    'description': 'High-quality plant image validated',\n                    'season': ''\n                })\n            \n            plants_found.append(plant_info)\n        \n        # Sort by validation score (highest first)\n        plants_found.sort(key=lambda x: x['validation_score'], reverse=True)\n        \n        # Generate statistics\n        plant_types = [p.get('plant_type', 'Unknown') for p in plants_found if p.get('plant_type') != 'Unknown']\n        type_counts = {}\n        for plant_type in plant_types:\n            type_counts[plant_type] = type_counts.get(plant_type, 0) + 1\n        \n        summary = {\n            'total_plants_found': len(plants_found),\n            'plants_with_identification': len(plant_identifications),\n            'plant_type_distribution': type_counts,\n            'average_validation_score': (\n                sum(p['validation_score'] for p in plants_found) / max(1, len(plants_found))\n            ),\n            'plants': plants_found,\n            'generation_timestamp': time.time()\n        }\n        \n        return summary\n    \n    def get_metrics(self) -> Dict[str, Any]:\n        """Get processing metrics for this stage.\n        \n        Returns:\n            Dictionary of metrics\n        """\n        return {\n            'input_images': self.processed_images,\n            'validated_images': self.validated_images,\n            'validation_pass_rate': self.validated_images / max(1, self.processed_images),\n            'api_success_rate': self.api_stats['successful_calls'] / max(1, self.api_stats['total_calls']),\n            'total_api_calls': self.api_stats['total_calls'],\n            'successful_api_calls': self.api_stats['successful_calls'],\n            'failed_api_calls': self.api_stats['failed_calls'],\n            'min_validation_score': self.min_validation_score\n        }