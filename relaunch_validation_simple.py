#!/usr/bin/env python3
"""Simple script to relaunch final validation stage without complex imports."""

import json
import sys
import os
from pathlib import Path

def main():
    """Run final validation using the main pipeline with a custom stage runner."""
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python3 relaunch_validation_simple.py <quality_results_path> <output_dir> [config_path]")
        print("Example: python3 relaunch_validation_simple.py /home/lounes/turf2/results/run_1753917646/stage_quality_assessment/quality_assessment_results.json ./final_validation_output")
        sys.exit(1)
    
    quality_results_path = sys.argv[1]
    output_dir = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Verify input file exists
    if not os.path.exists(quality_results_path):
        print(f"Error: Quality results file not found: {quality_results_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading quality results from: {quality_results_path}")
    print(f"Output directory: {output_dir}")
    
    # Load quality results
    try:
        with open(quality_results_path, 'r') as f:
            quality_data = json.load(f)
    except Exception as e:
        print(f"Error loading quality results: {e}")
        sys.exit(1)
    
    high_quality_crops = quality_data.get('high_quality_crops', [])
    print(f"Found {len(high_quality_crops)} high quality crops")
    
    if not high_quality_crops:
        print("No high quality crops found!")
        sys.exit(1)
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        # Import the modules we need by adding current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Now import after path is set
        from src.stages.stage6_validation import FinalValidationStage
        from src.utils import get_config, setup_logger
        
        # Setup logging
        logger = setup_logger('FinalValidationRelaunch', 'INFO')
        
        # Load config
        config = get_config(config_path)
        
        # Initialize and run validation stage
        logger.info("Initializing FinalValidationStage")
        validation_stage = FinalValidationStage(config.config)
        
        # Override the validation prompt to be less restrictive
        new_prompt = (
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

        # Override the prompt and lower the minimum score
        validation_stage.validation_prompt = new_prompt
        validation_stage.min_validation_score = 50  # Lower threshold
        
        logger.info("Starting final validation process")
        results = validation_stage.process(quality_data, output_dir)
        
        # Print results
        validated_count = len(results.get('validated_crops', []))
        total_count = len(results.get('validation_results', {}))
        summary = results.get('summary', {})
        
        print("\n=== FINAL VALIDATION COMPLETE ===")
        print(f"Plants validated: {validated_count}/{total_count}")
        print(f"Plants with identification: {summary.get('plants_with_identification', 0)}")
        print(f"Results saved to: {output_dir}")
        
        # Print plant types
        plant_types = summary.get('plant_type_distribution', {})
        if plant_types:
            print("\nPlant types detected:")
            for plant_type, count in plant_types.items():
                print(f"  - {plant_type}: {count}")
        
        return 0
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory and all dependencies are installed")
        return 1
    except Exception as e:
        print(f"Error during validation: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())