import cv2
import numpy as np
import os

def calculate_blur_score(image):
    """Calculate blur score using variance of Laplacian"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Laplacian operator and calculate variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def process_video_blur_detection(video_path, output_dir="blur_detection_output", blur_threshold=100.0):
    """
    Process video for blur detection and save frames with scores
    
    Args:
        video_path: Path to input MP4 video
        output_dir: Directory to save output frames
        blur_threshold: Threshold below which frames are considered blurry
    """
    
    # Create output directories
    sharp_dir = os.path.join(output_dir, "sharp_frames")
    blurry_dir = os.path.join(output_dir, "blurry_frames")
    
    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(blurry_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Blur threshold: {blur_threshold}")
    print("-" * 50)
    
    frame_count = 0
    sharp_count = 0
    blurry_count = 0
    
    blur_scores = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate blur score
        blur_score = calculate_blur_score(frame)
        blur_scores.append(blur_score)
        
        # Determine if frame is blurry
        is_blurry = blur_score < blur_threshold
        
        # Save frame to appropriate directory (no text overlay)
        timestamp = frame_count / fps
        filename = f"frame_{frame_count:06d}_t{timestamp:.2f}s_score{blur_score:.2f}.jpg"
        
        if is_blurry:
            output_path = os.path.join(blurry_dir, filename)
            blurry_count += 1
        else:
            output_path = os.path.join(sharp_dir, filename)
            sharp_count += 1
        
        # Save original frame without any modifications
        cv2.imwrite(output_path, frame)
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    
    # Calculate statistics
    blur_scores = np.array(blur_scores)
    mean_score = np.mean(blur_scores)
    std_score = np.std(blur_scores)
    min_score = np.min(blur_scores)
    max_score = np.max(blur_scores)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Sharp frames: {sharp_count} ({sharp_count/frame_count*100:.1f}%)")
    print(f"Blurry frames: {blurry_count} ({blurry_count/frame_count*100:.1f}%)")
    print(f"\nBlur Score Statistics:")
    print(f"  Mean: {mean_score:.2f}")
    print(f"  Std:  {std_score:.2f}")
    print(f"  Min:  {min_score:.2f}")
    print(f"  Max:  {max_score:.2f}")
    print(f"\nOutput directories:")
    print(f"  Sharp frames: {sharp_dir}")
    print(f"  Blurry frames: {blurry_dir}")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, "blur_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Blur threshold: {blur_threshold}\n")
        f.write(f"Total frames: {frame_count}\n")
        f.write(f"Sharp frames: {sharp_count} ({sharp_count/frame_count*100:.1f}%)\n")
        f.write(f"Blurry frames: {blurry_count} ({blurry_count/frame_count*100:.1f}%)\n")
        f.write(f"Mean blur score: {mean_score:.2f}\n")
        f.write(f"Std blur score: {std_score:.2f}\n")
        f.write(f"Min blur score: {min_score:.2f}\n")
        f.write(f"Max blur score: {max_score:.2f}\n")
    
    print(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    VIDEO_PATH = "../turf2/data/20250730_183739.mp4" 
    OUTPUT_DIR = "blur_detection_results"
    BLUR_THRESHOLD = 175.0  
    
    # Process the video
    process_video_blur_detection(VIDEO_PATH, OUTPUT_DIR, BLUR_THRESHOLD)