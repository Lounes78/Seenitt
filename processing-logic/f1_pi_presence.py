import cv2
import numpy as np
import os
import yaml
from ultralytics import YOLO
import shutil
from pathlib import Path

def load_plant_classes(yaml_path):
    """Load plant class IDs from YAML config file"""
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            return set(config['classes'])
    except Exception as e:
        print(f"Error loading YAML config: {e}")
        return set()

def detect_plants(image, results, plant_classes, class_names):
    """Check for plant detections and return count and detected plant types"""
    plant_count = 0
    detected_plants = []
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if class_id in plant_classes:
                    plant_count += 1
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    detected_plants.append(class_name)
    
    return plant_count, detected_plants

def process_plant_detection(sharp_frames_dir, output_dir, config_path, confidence_threshold=0.25):
    """
    Process sharp frames for plant detection and save frames containing plants
    
    Args:
        sharp_frames_dir: Directory containing sharp frames from blur detection
        output_dir: Directory to save frames with plant detections
        config_path: Path to YAML config with plant class IDs
        confidence_threshold: Minimum confidence for plant detections
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    plant_classes = load_plant_classes(config_path)
    if not plant_classes:
        print(f"Warning: No plant classes loaded from {config_path}")
        return
    
    print(f"Loaded {len(plant_classes)} plant class IDs: {sorted(list(plant_classes))}")
    
    print("Loading YOLOv8 model with OpenImages V7 weights...")
    try:
        model = YOLO('yolov8x-oiv7.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying with smaller model...")
        try:
            model = YOLO('yolov8n-oiv7.pt')
        except Exception as e2:
            print(f"Error loading smaller model: {e2}")
            return
    
    class_names = model.names if hasattr(model, 'names') else {}
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for file in os.listdir(sharp_frames_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"No image files found in {sharp_frames_dir}")
        return
    
    print(f"Processing {len(image_files)} sharp frames...")
    print(f"Confidence threshold: {confidence_threshold}")
    print("-" * 60)
    
    total_processed = 0
    frames_with_plants = 0
    total_plant_detections = 0
    plant_type_counts = {}
    
    for filename in sorted(image_files):
        input_path = os.path.join(sharp_frames_dir, filename)
        
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Could not load {filename}")
            continue
        
        total_processed += 1
        
        results = model(image, conf=confidence_threshold, verbose=False)
        
        plant_count, detected_plants = detect_plants(image, results, plant_classes, class_names)
        
        if plant_count > 0:
            frames_with_plants += 1
            total_plant_detections += plant_count
            
            for plant_type in detected_plants:
                plant_type_counts[plant_type] = plant_type_counts.get(plant_type, 0) + 1
            
            base_name = Path(filename).stem
            output_filename = f"{base_name}_plants{plant_count}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, image)
            
            print(f"✓ {filename} -> {plant_count} plants detected: {', '.join(set(detected_plants))}")
        
        if total_processed % 50 == 0:
            progress = (total_processed / len(image_files)) * 100
            print(f"Processed {total_processed}/{len(image_files)} frames ({progress:.1f}%)")
    
    plant_detection_rate = (frames_with_plants / total_processed * 100) if total_processed > 0 else 0
    avg_plants_per_frame = (total_plant_detections / frames_with_plants) if frames_with_plants > 0 else 0
    
    print("\n" + "="*70)
    print("PLANT DETECTION COMPLETE")
    print("="*70)
    print(f"Total frames processed: {total_processed}")
    print(f"Frames with plants: {frames_with_plants} ({plant_detection_rate:.1f}%)")
    print(f"Total plant detections: {total_plant_detections}")
    print(f"Average plants per frame: {avg_plants_per_frame:.1f}")
    
    if plant_type_counts:
        print(f"\nDetected plant types:")
        for plant_type, count in sorted(plant_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {plant_type}: {count} detections")
    
    print(f"\nOutput directory: {output_dir}")
    
    stats_file = os.path.join(output_dir, "plant_detection_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Plant Detection Results\n")
        f.write(f"======================\n\n")
        f.write(f"Input directory: {sharp_frames_dir}\n")
        f.write(f"Config file: {config_path}\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write(f"Plant classes used: {sorted(list(plant_classes))}\n\n")
        f.write(f"Total frames processed: {total_processed}\n")
        f.write(f"Frames with plants: {frames_with_plants} ({plant_detection_rate:.1f}%)\n")
        f.write(f"Total plant detections: {total_plant_detections}\n")
        f.write(f"Average plants per frame: {avg_plants_per_frame:.1f}\n\n")
        
        if plant_type_counts:
            f.write("Detected plant types:\n")
            for plant_type, count in sorted(plant_type_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {plant_type}: {count} detections\n")
    
    print(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    SHARP_FRAMES_DIR = "/home/lounes/turf3/blur_detection_results/sharp_frames"
    OUTPUT_DIR = "/home/lounes/turf3/frames_with_plants"
    CONFIG_PATH = "/home/lounes/turf2/config/filter_classes_OpenImagesV7.yaml"
    CONFIDENCE_THRESHOLD = 0.25
    
    if not os.path.exists(SHARP_FRAMES_DIR):
        print(f"Error: Input directory {SHARP_FRAMES_DIR} does not exist")
        print("Make sure to run the blur detection script first")
    elif not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file {CONFIG_PATH} does not exist")
        print("Please create the YAML config file with plant class IDs")
    else:
        process_plant_detection(SHARP_FRAMES_DIR, OUTPUT_DIR, CONFIG_PATH, CONFIDENCE_THRESHOLD)