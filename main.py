#!/usr/bin/env python3
"""
Video Processing Pipeline
Integrates blur detection, plant detection, and similarity analysis
to extract the best frames from a video containing plants.

Usage:
    python main.py <video_path> [output_folder]
"""

import sys
import os
import argparse
from pathlib import Path
import tempfile
import shutil
import time

from f0_blur_score import process_video_blur_detection
from f1_pi_presence import process_plant_detection
from f2_sim_GPU import analyze_image_similarity
from f3_qwen_judge import qwen_judge

def create_output_structure(output_dir):
    """Create organized output directory structure"""
    dirs = {
        'blur_detection': os.path.join(output_dir, '01_blur_detection'),
        'plant_detection': os.path.join(output_dir, '02_plant_detection'), 
        'similarity_analysis': os.path.join(output_dir, '03_similarity_analysis'),
        'qwen_analysis': os.path.join(output_dir, '04_qwen_analysis'),
        'final_frames': os.path.join(output_dir, '05_final_frames')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def validate_inputs(video_path, config_path):
    """Validate input files exist"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    video_ext = Path(video_path).suffix.lower()
    if video_ext not in ['.mp4', '.avi', '.mov', '.mkv']:
        raise ValueError(f"Unsupported video format: {video_ext}")


def process_video_pipeline(video_path, output_dir, config_path=None, 
                          blur_threshold=175.0, confidence_threshold=0.25, 
                          similarity_threshold=0.88):
    """
    Complete video processing pipeline
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save all outputs
        config_path: Path to YAML config with plant class IDs
        blur_threshold: Threshold for blur detection
        confidence_threshold: Threshold for plant detection confidence
        similarity_threshold: Threshold for similarity deduplication
    
    Returns:
        Path to final frames directory
    """
    
    # Set default config path if not provided
    if config_path is None:
        config_path = "/home/lounes/turf2/config/filter_classes_OpenImagesV7.yaml"
    
    # Validate inputs
    validate_inputs(video_path, config_path)
    
    # Create output structure
    dirs = create_output_structure(output_dir)
    
    print("="*80)
    print("VIDEO PROCESSING PIPELINE STARTING")
    print("="*80)
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {config_path}")
    print(f"Blur threshold: {blur_threshold}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Similarity threshold: {similarity_threshold}")
    print("="*80)
    
    try:
        # Step 1: Blur Detection
        start_time = time.time()

        print("\nSTEP 1: BLUR DETECTION")
        print("-" * 40)
        process_video_blur_detection(
            video_path=video_path,
            output_dir=dirs['blur_detection'],
            blur_threshold=blur_threshold
        )
        
        sharp_frames_dir = os.path.join(dirs['blur_detection'], 'sharp_frames')
        if not os.path.exists(sharp_frames_dir) or not os.listdir(sharp_frames_dir):
            raise RuntimeError("No sharp frames found after blur detection")
        
        print(f"\n✓ Blur detection complete. Sharp frames: {sharp_frames_dir}")
        
        end_phase = time.time()
        blur_phase1_duration = end_phase - start_time        
        
        # Step 2: Plant Detection
        print("\nSTEP 2: PLANT DETECTION")
        print("-" * 40)
        process_plant_detection(
            sharp_frames_dir=sharp_frames_dir,
            output_dir=dirs['plant_detection'],
            config_path=config_path,
            confidence_threshold=confidence_threshold
        )
        
        if not os.path.exists(dirs['plant_detection']) or not os.listdir(dirs['plant_detection']):
            raise RuntimeError("No frames with plants found after plant detection")
        
        print(f"\n✓ Plant detection complete. Frames with plants: {dirs['plant_detection']}")
        

        end_phase2 = time.time()
        plant_detec_phase2_duration = end_phase2 - end_phase  

        # Step 3: Similarity Analysis and Deduplication
        print("\nSTEP 3: SIMILARITY ANALYSIS & DEDUPLICATION")
        print("-" * 50)
        final_frames_dir = analyze_image_similarity(
            images_dir=dirs['plant_detection'],
            output_dir=dirs['similarity_analysis'],
            similarity_threshold=similarity_threshold
        )
        
        if not final_frames_dir or not os.path.exists(final_frames_dir):
            raise RuntimeError("Similarity analysis failed")
        
        print(f"\n✓ Similarity analysis complete. Deduplicated frames: {final_frames_dir}")
        
        end_phase3 = time.time()
        sim_phase3_duration = end_phase3 - end_phase2  


        # Step 4: Copy final frames to main output
        print("\nSTEP 4: VLM ANALYSIS")
        print("-" * 30)
        
        qwen_judge(image_dir=final_frames_dir, output_dir=dirs['qwen_analysis'])

        print(f"\n✓ Qwen analysis complete. Plant crops and analysis: {dirs['qwen_analysis']}")
        
        end_phase4 = time.time()
        qwen_phase4_duration = end_phase4 - end_phase3  

        end_total = time.time()
        total_duration = end_total - start_time

        # Create pipeline summary
        summary_file = os.path.join(output_dir, "pipeline_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("Video Processing Pipeline Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Input video: {video_path}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Config file: {config_path}\n\n")
            f.write("Processing parameters:\n")
            f.write(f"  Blur threshold: {blur_threshold}\n")
            f.write(f"  Confidence threshold: {confidence_threshold}\n")
            f.write(f"  Similarity threshold: {similarity_threshold}\n\n")
            f.write("Results:\n")
            f.write(f"  Qwen analysis location: {dirs['qwen_analysis']}\n\n")
            f.write("Processing stages:\n")
            f.write(f"  1. Blur detection: {dirs['blur_detection']}\n")
            f.write(f"  2. Plant detection: {dirs['plant_detection']}\n")
            f.write(f"  3. Similarity analysis: {dirs['similarity_analysis']}\n")
            f.write(f"  4. Qwen analysis: {dirs['qwen_analysis']}\n")
            f.write("\nDurations (in seconds):\n")
            f.write(f"  Blur detection: {blur_phase1_duration:.2f}\n")
            f.write(f"  Plant detection: {plant_detec_phase2_duration:.2f}\n")
            f.write(f"  Similarity analysis: {sim_phase3_duration:.2f}\n")
            f.write(f"  Qwen analysis: {qwen_phase4_duration:.2f}\n")
            f.write(f"  Total: {total_duration:.2f}\n")

        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)

        print(f"Phase 01: {blur_phase1_duration:.3f}")
        print(f"Phase 02: {plant_detec_phase2_duration:.3f}")
        print(f"Phase 03: {sim_phase3_duration:.3f}")
        print(f"Phase 04: {qwen_phase4_duration:.3f}")

        print(f"\nTotal pipeline duration: {total_duration:.3f} seconds")
        return dirs['qwen_analysis']
        
    except Exception as e:
        print(f"\nPipeline failed at current step: {str(e)}")
        raise


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Process video to extract best frames with plants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py video.mp4
    python main.py video.mp4 my_output_folder
    python main.py video.mp4 output --blur-threshold 200 --similarity-threshold 0.9
        """
    )
    
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('output_dir', nargs='?', 
                       help='Output directory (default: video_name_processed)')
    parser.add_argument('--config', 
                       default="/home/lounes/turf2/config/filter_classes_OpenImagesV7.yaml",
                       help='Path to plant classes config YAML file')
    parser.add_argument('--blur-threshold', type=float, default=175.0,
                       help='Blur detection threshold (default: 175.0)')
    parser.add_argument('--confidence-threshold', type=float, default=0.25,
                       help='Plant detection confidence threshold (default: 0.25)')
    parser.add_argument('--similarity-threshold', type=float, default=0.88,
                       help='Similarity deduplication threshold (default: 0.88)')
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        video_name = Path(args.video_path).stem
        args.output_dir = f"{video_name}_processed"
    
    try:
        final_dir = process_video_pipeline(
            video_path=args.video_path,
            output_dir=args.output_dir,
            config_path=args.config,
            blur_threshold=args.blur_threshold,
            confidence_threshold=args.confidence_threshold,
            similarity_threshold=args.similarity_threshold
        )
        
        print(f"\nSuccess! Final frames are in: {final_dir}")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())