"""Integration tests for the complete pipeline."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import cv2
import numpy as np

from src.pipeline import PlantRecognitionPipeline


class TestPipelineIntegration:
    """Integration tests for the complete processing pipeline."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return {
            'yolo': {
                'model_path': 'yolov8x-oiv7.pt',
                'confidence_threshold': 0.3,
                'device': 'cpu'
            },
            'filtering': {
                'plant_classes': [1, 2, 3],
                'min_detections_per_track': 2,
                'min_track_duration': 0.5
            },
            'quality': {
                'min_quality_score': 0.3,
                'enable_clip_quality': False  # Disable for testing
            },
            'similarity': {
                'threshold': 0.85
            },
            'validation': {
                'api_endpoint': '',  # Disable API calls for testing
                'min_validation_score': 70
            },
            'pipeline': {
                'save_intermediate_results': True
            }
        }
    
    @pytest.fixture
    def mock_video_file(self):
        """Create a mock video file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Create a simple test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f.name, fourcc, 1.0, (640, 480))
            
            # Write a few test frames
            for i in range(5):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            
            out.release()
            yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    def test_pipeline_initialization(self, test_config):
        """Test pipeline initialization with configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            pipeline = PlantRecognitionPipeline(config_path)
            
            # Check that all stages were initialized
            expected_stages = [
                'detection', 'tracking_filter', 'intra_track_filter',
                'similarity_detection', 'quality_assessment', 'final_validation'
            ]
            
            for stage in expected_stages:
                assert stage in pipeline.stages
            
            # Check configuration loading
            config = pipeline.get_configuration()
            assert config['yolo']['confidence_threshold'] == 0.3
            
        finally:
            Path(config_path).unlink()
    
    @patch('src.stages.stage1_detection.YOLO')
    def test_pipeline_stages_execution_order(self, mock_yolo, test_config, mock_video_file):
        """Test that pipeline stages execute in correct order."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_path = f.name
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                # Mock YOLO model
                mock_model = Mock()
                mock_yolo.return_value = mock_model
                
                # Mock YOLO tracking results
                mock_result = Mock()
                mock_result.boxes = None  # No detections for simplicity
                mock_model.track.return_value = [mock_result] * 3  # 3 frames
                
                # Initialize pipeline
                pipeline = PlantRecognitionPipeline(config_path)
                
                # Process video (this will run through all stages)
                results = pipeline.process_video(mock_video_file, output_dir)
                
                # Check that all stage results are present
                stage_results = results['stage_results']
                expected_stages = [
                    'detection', 'tracking_filter', 'intra_track_filter',
                    'similarity_detection', 'quality_assessment', 'final_validation'
                ]
                
                for stage in expected_stages:
                    assert stage in stage_results
                    assert 'stage_metrics' in stage_results[stage]
                
                # Check pipeline metrics
                assert 'pipeline_metrics' in results
                assert 'total_processing_time' in results
                assert 'summary' in results
                
            finally:
                Path(config_path).unlink()
    
    def test_pipeline_error_handling(self, test_config):
        """Test pipeline error handling with invalid inputs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_path = f.name
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                pipeline = PlantRecognitionPipeline(config_path)
                
                # Test with non-existent video file
                with pytest.raises(Exception):
                    pipeline.process_video('nonexistent_video.mp4', output_dir)
                
            finally:
                Path(config_path).unlink()
    
    def test_pipeline_output_structure(self, test_config, mock_video_file):
        """Test that pipeline creates expected output structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_path = f.name
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                with patch('src.stages.stage1_detection.YOLO') as mock_yolo:
                    # Mock YOLO model
                    mock_model = Mock()
                    mock_yolo.return_value = mock_model
                    mock_result = Mock()
                    mock_result.boxes = None
                    mock_model.track.return_value = [mock_result] * 2
                    
                    pipeline = PlantRecognitionPipeline(config_path)
                    results = pipeline.process_video(mock_video_file, output_dir)
                    
                    # Check that run directory was created
                    run_id = results['run_id']
                    run_dir = Path(output_dir) / run_id
                    assert run_dir.exists()
                    
                    # Check that stage directories were created
                    expected_dirs = [
                        'stage_detection', 'stage_tracking_filter', 'stage_intra_track_filter',
                        'stage_similarity_detection', 'stage_quality_assessment',
                        'stage_final_validation', 'summary'
                    ]
                    
                    for dir_name in expected_dirs:
                        stage_dir = run_dir / dir_name
                        assert stage_dir.exists()
                    
                    # Check that summary files were created
                    summary_dir = run_dir / 'summary'
                    assert (summary_dir / 'complete_pipeline_results.json').exists()
                    assert (summary_dir / 'plant_detection_summary.json').exists()
                    
            finally:
                Path(config_path).unlink()
    
    def test_configuration_update(self, test_config):
        """Test pipeline configuration updates."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            pipeline = PlantRecognitionPipeline(config_path)
            
            # Get original config
            original_config = pipeline.get_configuration()
            original_threshold = original_config['yolo']['confidence_threshold']
            
            # Update configuration
            new_config = {'yolo': {'confidence_threshold': 0.5}}
            pipeline.update_configuration(new_config)
            
            # Check that configuration was updated
            updated_config = pipeline.get_configuration()
            assert updated_config['yolo']['confidence_threshold'] == 0.5
            assert updated_config['yolo']['confidence_threshold'] != original_threshold
            
        finally:
            Path(config_path).unlink()