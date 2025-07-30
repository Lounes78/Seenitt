"""Main pipeline orchestrator for smart glasses plant recognition."""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from .utils import Config, get_config, setup_logger, ensure_dir
from .stages import (
    YOLODetectionStage,
    TrackingIDFilterStage,
    IntraTrackFilterStage,
    SimilarityDetectionStage,
    QualityAssessmentStage,
    FinalValidationStage
)


class PlantRecognitionPipeline:
    """Main pipeline for processing video streams and identifying plants."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = setup_logger('PlantRecognitionPipeline', 'INFO')
        
        # Pipeline configuration
        pipeline_config = self.config.get_pipeline_config()
        self.save_intermediate_results = pipeline_config.get('save_intermediate_results', True)
        self.cleanup_temp_files = pipeline_config.get('cleanup_temp_files', False)
        
        # Initialize stages
        self.stages = self._initialize_stages()
        
        # Pipeline state
        self.current_run_id = None
        self.run_start_time = None
        self.pipeline_results = {}
        
    def _initialize_stages(self) -> Dict[str, Any]:
        """Initialize all processing stages.
        
        Returns:
            Dictionary of initialized stage objects
        """
        config_dict = self.config.config
        
        stages = {
            'detection': YOLODetectionStage(config_dict),
            'tracking_filter': TrackingIDFilterStage(config_dict),
            'intra_track_filter': IntraTrackFilterStage(config_dict),
            'similarity_detection': SimilarityDetectionStage(config_dict),
            'quality_assessment': QualityAssessmentStage(config_dict),
            'final_validation': FinalValidationStage(config_dict)
        }
        
        self.logger.info(f"Initialized {len(stages)} processing stages")
        return stages
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID for this pipeline execution.
        
        Returns:
            Unique run identifier
        """
        timestamp = int(time.time())
        return f"run_{timestamp}"
    
    def _setup_output_directories(self, base_output_dir: str) -> Dict[str, str]:
        """Setup output directories for pipeline run.
        
        Args:
            base_output_dir: Base output directory
            
        Returns:
            Dictionary mapping stage names to output directories
        """
        run_dir = Path(base_output_dir) / self.current_run_id
        ensure_dir(run_dir)
        
        stage_dirs = {}
        for stage_name in self.stages.keys():
            stage_dir = run_dir / f"stage_{stage_name}"
            ensure_dir(stage_dir)
            stage_dirs[stage_name] = str(stage_dir)
        
        # Create summary directory
        summary_dir = run_dir / "summary"
        ensure_dir(summary_dir)
        stage_dirs['summary'] = str(summary_dir)
        
        return stage_dirs
    
    def process_video(self, video_path: str, output_dir: str) -> Dict[str, Any]:
        """Process a single video through the entire pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save pipeline outputs
            
        Returns:
            Dictionary containing complete pipeline results
        """
        self.current_run_id = self._generate_run_id()
        self.run_start_time = time.time()
        
        self.logger.info(f"Starting pipeline run {self.current_run_id} for video: {video_path}")
        
        # Setup output directories
        stage_dirs = self._setup_output_directories(output_dir)
        
        # Initialize pipeline results
        self.pipeline_results = {
            'run_id': self.current_run_id,
            'video_path': video_path,
            'start_time': self.run_start_time,
            'stage_results': {},
            'stage_metrics': {},
            'pipeline_metrics': {}
        }
        
        try:
            # Stage 1: YOLO Detection and Tracking
            self.logger.info("=== Stage 1: Object Detection and Tracking ===")
            stage1_start = time.time()
            stage1_results = self.stages['detection'].process(video_path, stage_dirs['detection'])
            stage1_time = time.time() - stage1_start
            
            self.pipeline_results['stage_results']['detection'] = stage1_results
            self.pipeline_results['stage_metrics']['detection'] = {
                'processing_time': stage1_time,
                **stage1_results.get('stage_metrics', {})
            }
            
            # Stage 2: Tracking ID Filtering
            self.logger.info("=== Stage 2: Tracking ID Filtering ===")
            stage2_start = time.time()
            stage2_results = self.stages['tracking_filter'].process(stage1_results, stage_dirs['tracking_filter'])
            stage2_time = time.time() - stage2_start
            
            self.pipeline_results['stage_results']['tracking_filter'] = stage2_results
            self.pipeline_results['stage_metrics']['tracking_filter'] = {
                'processing_time': stage2_time,
                **stage2_results.get('stage_metrics', {})
            }
            
            # Stage 3: Intra-Track Filtering (Best Crop Selection)
            self.logger.info("=== Stage 3: Intra-Track Filtering ===")
            stage3_start = time.time()
            stage3_results = self.stages['intra_track_filter'].process(stage2_results, stage_dirs['intra_track_filter'])
            stage3_time = time.time() - stage3_start
            
            self.pipeline_results['stage_results']['intra_track_filter'] = stage3_results
            self.pipeline_results['stage_metrics']['intra_track_filter'] = {
                'processing_time': stage3_time,
                **stage3_results.get('stage_metrics', {})
            }
            
            # Stage 4: Similarity Detection
            self.logger.info("=== Stage 4: Similarity Detection ===")
            stage4_start = time.time()
            stage4_results = self.stages['similarity_detection'].process(stage3_results, stage_dirs['similarity_detection'])
            stage4_time = time.time() - stage4_start
            
            self.pipeline_results['stage_results']['similarity_detection'] = stage4_results
            self.pipeline_results['stage_metrics']['similarity_detection'] = {
                'processing_time': stage4_time,
                **stage4_results.get('stage_metrics', {})
            }
            
            # Stage 5: Quality Assessment
            self.logger.info("=== Stage 5: Quality Assessment ===")
            stage5_start = time.time()
            stage5_results = self.stages['quality_assessment'].process(stage4_results, stage_dirs['quality_assessment'])
            stage5_time = time.time() - stage5_start
            
            self.pipeline_results['stage_results']['quality_assessment'] = stage5_results
            self.pipeline_results['stage_metrics']['quality_assessment'] = {
                'processing_time': stage5_time,
                **stage5_results.get('stage_metrics', {})
            }
            
            # Stage 6: Final Validation
            self.logger.info("=== Stage 6: Final Validation ===")
            stage6_start = time.time()
            stage6_results = self.stages['final_validation'].process(stage5_results, stage_dirs['final_validation'])
            stage6_time = time.time() - stage6_start
            
            self.pipeline_results['stage_results']['final_validation'] = stage6_results
            self.pipeline_results['stage_metrics']['final_validation'] = {
                'processing_time': stage6_time,
                **stage6_results.get('stage_metrics', {})
            }
            
            # Calculate pipeline metrics
            total_time = time.time() - self.run_start_time
            self.pipeline_results['end_time'] = time.time()
            self.pipeline_results['total_processing_time'] = total_time
            
            self.pipeline_results['pipeline_metrics'] = self._calculate_pipeline_metrics()
            
            # Generate final summary
            summary = self._generate_pipeline_summary()
            self.pipeline_results['summary'] = summary
            
            # Save complete pipeline results
            results_file = Path(stage_dirs['summary']) / 'complete_pipeline_results.json'
            with open(results_file, 'w') as f:
                json.dump(self.pipeline_results, f, indent=2)
            
            # Save user-friendly summary
            summary_file = Path(stage_dirs['summary']) / 'plant_detection_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            self.logger.info(f"Final result: {summary['total_plants_found']} plants detected and validated")
            
            return self.pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.pipeline_results['error'] = str(e)
            self.pipeline_results['status'] = 'failed'
            raise
    
    def process_live_stream(self, camera_index: int = 0, output_dir: str = "live_output") -> Dict[str, Any]:
        """Process live camera stream (placeholder for future implementation).
        
        Args:
            camera_index: Camera device index
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info("Live stream processing not yet implemented")
        # TODO: Implement live stream processing with frame buffering
        raise NotImplementedError("Live stream processing will be implemented in future version")
    
    def _calculate_pipeline_metrics(self) -> Dict[str, Any]:
        """Calculate overall pipeline performance metrics.
        
        Returns:
            Dictionary of pipeline metrics
        """
        stage_metrics = self.pipeline_results['stage_metrics']
        
        # Processing time breakdown
        time_breakdown = {}
        total_stage_time = 0
        for stage, metrics in stage_metrics.items():
            stage_time = metrics.get('processing_time', 0)
            time_breakdown[stage] = stage_time
            total_stage_time += stage_time
        
        # Data flow metrics
        detection_results = self.pipeline_results['stage_results']['detection']
        final_results = self.pipeline_results['stage_results']['final_validation']
        
        initial_detections = detection_results.get('detection_results', {}).get('total_detections', 0)
        final_plants = len(final_results.get('validated_crops', []))
        
        # Calculate reduction rates through pipeline
        stages_data = [
            ('initial_detections', initial_detections),
            ('filtered_detections', detection_results.get('detection_results', {}).get('filtered_detections', 0)),
            ('valid_tracks', len(self.pipeline_results['stage_results']['tracking_filter'].get('valid_tracks', []))),
            ('selected_crops', len(self.pipeline_results['stage_results']['intra_track_filter'].get('selected_crops', []))),
            ('unique_crops', len(self.pipeline_results['stage_results']['similarity_detection'].get('unique_crops', []))),
            ('high_quality_crops', len(self.pipeline_results['stage_results']['quality_assessment'].get('high_quality_crops', []))),
            ('validated_plants', final_plants)
        ]
        
        return {
            'total_processing_time': self.pipeline_results['total_processing_time'],
            'time_breakdown': time_breakdown,
            'data_flow': dict(stages_data),
            'pipeline_efficiency': final_plants / max(1, initial_detections),
            'plants_per_second': final_plants / max(1, self.pipeline_results['total_processing_time']),
            'avg_time_per_stage': total_stage_time / len(stage_metrics)
        }
    
    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate user-friendly pipeline summary.
        
        Returns:
            Summary dictionary for end users
        """
        final_results = self.pipeline_results['stage_results']['final_validation']
        summary_data = final_results.get('summary', {})
        
        # Add pipeline context
        pipeline_summary = {
            'session_info': {
                'run_id': self.current_run_id,
                'video_processed': self.pipeline_results['video_path'],
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.run_start_time)),
                'total_processing_time': f"{self.pipeline_results['total_processing_time']:.2f} seconds"
            },
            'detection_summary': {
                'total_plants_found': summary_data.get('total_plants_found', 0),
                'plants_with_identification': summary_data.get('plants_with_identification', 0),
                'plant_types_detected': summary_data.get('plant_type_distribution', {}),
                'average_confidence': summary_data.get('average_validation_score', 0)
            },
            'pipeline_performance': self.pipeline_results['pipeline_metrics'],
            'detected_plants': summary_data.get('plants', [])
        }
        
        return pipeline_summary
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current pipeline configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.config
    
    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """Update pipeline configuration (requires reinitialization).
        
        Args:
            new_config: New configuration dictionary
        """
        # Save current config
        old_config = self.config.config.copy()
        
        try:
            # Update config
            self.config._config.update(new_config)
            
            # Reinitialize stages with new config
            self.stages = self._initialize_stages()
            
            self.logger.info("Pipeline configuration updated successfully")
            
        except Exception as e:
            # Restore old config on failure
            self.config._config = old_config
            self.logger.error(f"Failed to update configuration: {e}")
            raise


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Glasses Plant Recognition Pipeline')
    parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger('Main', log_level)
    
    try:
        # Initialize pipeline
        pipeline = PlantRecognitionPipeline(args.config)
        
        # Process video
        results = pipeline.process_video(args.video, args.output)
        
        # Print summary
        summary = results['summary']
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Plants detected: {summary['detection_summary']['total_plants_found']}")
        logger.info(f"Processing time: {summary['session_info']['total_processing_time']}")
        logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())