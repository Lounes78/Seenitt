# Smart Glasses Plant Recognition System

A comprehensive computer vision pipeline for real-time plant and tree identification through smart glasses, featuring multi-stage filtering and AI-powered validation.

## Overview

This system processes live video streams from smart glasses to automatically identify and catalog plants and trees. The end result is a comprehensive summary delivered to the user's phone, featuring images and names of all vegetation encountered during walks or exploration.

## System Architecture

### Multi-Stage Processing Pipeline

1. **Stage 1: Object Detection & Tracking**
   - YOLO-based detection and tracking of vegetation
   - Bounding box generation and tracking ID assignment
   - Initial filtering by class and size criteria

2. **Stage 2: Tracking ID Filtering**
   - Filter tracks based on consistency and reliability
   - Minimum detection count, duration, and confidence thresholds
   - Bounding box stability analysis

3. **Stage 3: Intra-Track Filtering**
   - Select the best crop from each tracking ID
   - Quality-based selection using CLIP metrics
   - Composite scoring (quality + confidence + centrality + area)

4. **Stage 4: Similarity Detection**
   - Remove duplicates across different tracking IDs
   - CLIP embedding-based similarity analysis
   - Spatial and temporal constraints

5. **Stage 5: Quality Assessment**
   - Multi-criteria quality evaluation
   - CLIP quality scoring, blur detection, brightness/contrast analysis
   - Resolution and file integrity checks

6. **Stage 6: Final Validation**
   - AI-powered validation using Qwen 2.5VL API
   - Plant identification and classification
   - Final quality scoring and botanical suitability assessment

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd smart_glasses_plant_recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO model:**
   ```bash
   # Download YOLOv8x trained on Open Images V7
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt
   ```

### Basic Usage

```bash
# Process a video file
python main.py --video data/walk_video.mp4 --output results/

# With verbose logging
python main.py --video data/walk_video.mp4 --output results/ --verbose

# With custom configuration
python main.py --video data/walk_video.mp4 --config custom_config.yaml --output results/
```

### Configuration

Edit `config/config.yaml` to customize processing parameters:

```yaml
# YOLO Detection Settings
yolo:
  model_path: "yolov8x-oiv7.pt"
  confidence_threshold: 0.3
  device: "cuda"  # or "cpu"

# Class Filtering
filtering:
  filter_classes_file: "config/filter_classes_OpenImagesV7.yaml"
  # plant_classes: []  # Override filter file with specific classes
  min_bbox_area: 1000
  min_detections_per_track: 3

# Quality Assessment
quality:
  min_quality_score: 0.3
  enable_clip_quality: true
  enable_blur_detection: true

# Similarity Detection
similarity:
  threshold: 0.85
  use_spatial_constraint: true

# Final Validation API
validation:
  api_endpoint: "https://your-qwen-endpoint.com/vision"
  min_validation_score: 70
  enable_plant_identification: true
```

### Plant Class Filtering

The system uses `config/filter_classes_OpenImagesV7.yaml` to define which plant classes to detect:

- **45 plant classes** including fruits, vegetables, trees, flowers, and houseplants
- **Organized by category** for easy customization
- **Custom filters** can be created for specific use cases (orchards, gardens, forests)

To create a custom filter:
1. Copy `config/example_custom_filter.yaml`
2. Modify the classes list for your needs
3. Update `filter_classes_file` in `config.yaml`

## Project Structure

```
smart_glasses_plant_recognition/
├── src/
│   ├── stages/              # Processing stage modules
│   │   ├── stage1_detection.py
│   │   ├── stage2_filtering.py
│   │   ├── stage3_intra_track.py
│   │   ├── stage4_similarity.py
│   │   ├── stage5_quality.py
│   │   └── stage6_validation.py
│   ├── utils/               # Utility modules
│   │   ├── config.py
│   │   ├── logging_utils.py
│   │   └── file_utils.py
│   └── pipeline.py          # Main pipeline orchestrator
├── config/
│   └── config.yaml          # Configuration file
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── data/                    # Input data directory
├── output/                  # Output results directory
├── main.py                  # Main entry point
└── requirements.txt         # Python dependencies
```

## 🔧 Pipeline Stages Details

### Stage 1: YOLO Detection & Tracking
- **Input:** Video stream
- **Output:** Detected objects with tracking IDs, cropped images
- **Key Features:**
  - Real-time object detection and tracking
  - Configurable confidence thresholds
  - Plant/vegetation class filtering
  - Automatic image cropping

### Stage 2: Tracking ID Filtering
- **Input:** Detection results and crops
- **Output:** Filtered tracks based on reliability
- **Filtering Criteria:**
  - Minimum detections per track
  - Track duration thresholds
  - Confidence consistency
  - Bounding box stability (IoU-based)

### Stage 3: Intra-Track Filtering
- **Input:** Filtered crops grouped by track ID
- **Output:** Best quality crop per track
- **Selection Criteria:**
  - CLIP-based quality assessment
  - Detection confidence
  - Object centrality in frame
  - Bounding box area

### Stage 4: Similarity Detection
- **Input:** Selected crops from each track
- **Output:** Unique crops with duplicates removed
- **Features:**
  - CLIP embedding similarity
  - Configurable similarity threshold
  - Spatial distance constraints
  - Temporal proximity filtering

### Stage 5: Quality Assessment
- **Input:** Unique crops
- **Output:** High-quality images suitable for identification
- **Quality Metrics:**
  - CLIP quality assessment
  - Blur detection (Laplacian variance)
  - Brightness and contrast analysis
  - Resolution requirements

### Stage 6: Final Validation
- **Input:** High-quality crops
- **Output:** Validated plants with identification
- **AI Services:**
  - Qwen 2.5VL API integration
  - Plant suitability validation
  - Species identification
  - Confidence scoring

## Output Format

The pipeline generates comprehensive results:

### Plant Summary (`plant_detection_summary.json`)
```json
{
  "session_info": {
    "run_id": "run_1234567890",
    "video_processed": "data/walk_video.mp4",
    "processing_date": "2024-01-15 14:30:00",
    "total_processing_time": "45.67 seconds"
  },
  "detection_summary": {
    "total_plants_found": 12,
    "plants_with_identification": 8,
    "plant_types_detected": {
      "tree": 5,
      "flower": 3,
      "shrub": 2
    },
    "average_confidence": 85.3
  },
  "detected_plants": [
    {
      "track_id": 1,
      "common_name": "Red Maple",
      "scientific_name": "Acer rubrum",
      "plant_type": "tree",
      "identification_confidence": 92,
      "validation_score": 88,
      "image_path": "output/run_1234567890/stage_final_validation/validated_crops/validated_track_1.jpg"
    }
  ]
}
```

### Directory Structure
```
output/run_1234567890/
├── stage_detection/
│   ├── crops/              # All detected crops
│   └── detection_tracking_results.json
├── stage_tracking_filter/
│   ├── filtered_crops/     # Filtered by tracking criteria
│   └── tracking_id_filter_results.json
├── stage_intra_track_filter/
│   ├── selected_crops/     # Best crop per track
│   └── intra_track_filter_results.json
├── stage_similarity_detection/
│   ├── unique_crops/       # Duplicates removed
│   └── similarity_detection_results.json
├── stage_quality_assessment/
│   ├── high_quality_crops/ # Quality filtered
│   └── quality_assessment_results.json
├── stage_final_validation/
│   ├── validated_crops/    # Final validated plants
│   └── final_validation_results.json
└── summary/
    ├── complete_pipeline_results.json
    └── plant_detection_summary.json
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src
```
