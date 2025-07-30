#!/usr/bin/env python3
"""Main entry point for the Smart Glasses Plant Recognition application."""

import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pipeline import PlantRecognitionPipeline
from src.utils import setup_logger


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Smart Glasses Plant Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video data/walk_video.mp4 --output results/
  python main.py --video data/walk_video.mp4 --config custom_config.yaml --verbose
  python main.py --help
        """
    )
    
    parser.add_argument(
        '--video', '-v',
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--show-display',
        action='store_true',
        help='Show real-time detection display during processing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger('SmartGlassesPlantRecognition', log_level)
    
    logger.info("=== Smart Glasses Plant Recognition System ===")
    logger.info(f"Video input: {args.video}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Configuration: {args.config or 'default'}")
    
    try:
        # Validate input video exists
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Input video file not found: {args.video}")
            return 1
        
        # Initialize pipeline
        logger.info("Initializing processing pipeline...")
        pipeline = PlantRecognitionPipeline(args.config)
        
        # Process video
        logger.info("Starting video processing...")
        results = pipeline.process_video(str(video_path), args.output)
        
        # Display results summary
        summary = results['summary']
        detection_summary = summary['detection_summary']
        session_info = summary['session_info']
        
        logger.info("=== PROCESSING COMPLETE ===")
        logger.info(f"Plants detected and validated: {detection_summary['total_plants_found']}")
        logger.info(f"Plants with identification: {detection_summary['plants_with_identification']}")
        logger.info(f"Processing time: {session_info['total_processing_time']}")
        logger.info(f"Results saved to: {args.output}")
        
        # Show plant types if any were identified
        plant_types = detection_summary.get('plant_types_detected', {})
        if plant_types:
            logger.info("Plant types detected:")
            for plant_type, count in plant_types.items():
                logger.info(f"  - {plant_type}: {count}")
        
        # Show pipeline efficiency
        pipeline_metrics = summary['pipeline_performance']
        efficiency = pipeline_metrics.get('pipeline_efficiency', 0) * 100
        logger.info(f"Pipeline efficiency: {efficiency:.1f}% (valid plants / total detections)")
        
        logger.info("Check the output directory for detailed results and plant images!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit(main())