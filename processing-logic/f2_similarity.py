import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import shutil
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

def extract_blur_score(filename):
    """Extract blur score from filename"""
    try:
        # Format: frame_000001_t0.03s_score156.45_plants2.jpg
        if 'score' in filename:
            score_part = filename.split('score')[1]
            # Get the number before the next underscore or before _plants
            if '_plants' in score_part:
                score_str = score_part.split('_plants')[0]
            elif '_' in score_part:
                score_str = score_part.split('_')[0]
            else:
                score_str = score_part.split('.')[0]  # Remove extension
            return float(score_str)
        return 0.0
    except Exception as e:
        print(f"Warning: Could not extract blur score from {filename}: {e}")
        return 0.0



def compute_similarity_matrix(embeddings, device):
    "done on gpu"
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # Normalization, they should be already normalized but just in case # testing
    embeddings_normalized = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)

    similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())

    similarity_matrix_cpu = similarity_matrix.cpu().numpy()

    print(f"Similarity matrix computed on gpu, shape: {similarity_matrix_cpu.shape}")

    return similarity_matrix_cpu



def load_and_process_images(image_dir, clip_model, clip_processor, device):
    """Load images and compute CLIP embeddings"""
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for file in os.listdir(image_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return None, None
    
    print(f"Processing {len(image_files)} images...")
    
    embeddings = []
    valid_filenames = []
    
    for i, filename in enumerate(sorted(image_files)):
        try:
            # Load image
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert('RGB')
            
            # Process image through CLIP
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            
            # Get image embeddings
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu().numpy().flatten())
            valid_filenames.append(filename)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    if not embeddings:
        print("No valid embeddings generated")
        return None, None
    
    embeddings_array = np.array(embeddings)
    print(f"Generated embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array, valid_filenames





def deduplicate_similar_images(similarity_matrix, filenames, images_dir, similarity_threshold=0.95):
    """
    Remove highly similar images, keeping the one with higher blur score (sharper image)
    
    Args:
        similarity_matrix: Cosine similarity matrix
        filenames: List of image filenames
        images_dir: Directory containing the images
        similarity_threshold: Threshold above which images are considered duplicates
    
    Returns:
        List of filenames to keep
    """
    
    print(f"\nRemoving highly similar images (similarity > {similarity_threshold})...")
    
    n = len(similarity_matrix)
    images_to_remove = set()
    similar_pairs = []
    
    # Find all similar pairs
    for i in range(n):
        for j in range(i+1, n):
            similarity = similarity_matrix[i][j]
            if similarity > similarity_threshold:
                similar_pairs.append((i, j, filenames[i], filenames[j], similarity))
    
    print(f"Found {len(similar_pairs)} highly similar pairs")
    
    # For each similar pair, decide which one to remove
    for i, j, file1, file2, similarity in similar_pairs:
        # Skip if one of the images is already marked for removal
        if file1 in images_to_remove or file2 in images_to_remove:
            continue
        
        # Extract blur scores
        blur_score1 = extract_blur_score(file1)
        blur_score2 = extract_blur_score(file2)
        
        # Keep the image with higher blur score (sharper)
        if blur_score1 > blur_score2:
            images_to_remove.add(file2)
            print(f"  Removing {file2} (blur: {blur_score2:.2f}) - keeping {file1} (blur: {blur_score1:.2f}) [sim: {similarity:.4f}]")
        else:
            images_to_remove.add(file1)
            print(f"  Removing {file1} (blur: {blur_score1:.2f}) - keeping {file2} (blur: {blur_score2:.2f}) [sim: {similarity:.4f}]")
    
    # Create list of images to keep
    images_to_keep = [f for f in filenames if f not in images_to_remove]
    
    print(f"\nDeduplication summary:")
    print(f"  Original images: {len(filenames)}")
    print(f"  Images to remove: {len(images_to_remove)}")
    print(f"  Images to keep: {len(images_to_keep)}")
    
    return images_to_keep, images_to_remove



def create_deduplicated_folder(images_dir, images_to_keep, output_dir):
    """Copy deduplicated images to a new folder"""
    
    deduplicated_dir = os.path.join(output_dir, "deduplicated_frames")
    os.makedirs(deduplicated_dir, exist_ok=True)
    
    print(f"\nCreating deduplicated folder: {deduplicated_dir}")
    
    copied_count = 0
    for filename in images_to_keep:
        src_path = os.path.join(images_dir, filename)
        dst_path = os.path.join(deduplicated_dir, filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {filename}: {e}")
    
    print(f"Successfully copied {copied_count} deduplicated images")
    return deduplicated_dir



def print_similarity_stats(similarity_matrix, filenames):
    """Print similarity statistics and top similar pairs"""
    
    print("\n" + "="*60)
    print("SIMILARITY ANALYSIS RESULTS")
    print("="*60)
    
    # Get upper triangle (excluding diagonal)
    n = len(similarity_matrix)
    similarities = []
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            similarity = similarity_matrix[i][j]
            similarities.append(similarity)
            pairs.append((filenames[i], filenames[j], similarity))
    
    similarities = np.array(similarities)
    
    print(f"Number of image pairs: {len(similarities)}")
    print(f"Mean similarity: {similarities.mean():.4f}")
    print(f"Std similarity: {similarities.std():.4f}")
    print(f"Min similarity: {similarities.min():.4f}")
    print(f"Max similarity: {similarities.max():.4f}")
    
    # Sort pairs by similarity
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTOP 10 MOST SIMILAR PAIRS:")
    print("-" * 60)
    for i, (file1, file2, sim) in enumerate(pairs[:10], 1):
        # Shorten filenames for display
        short1 = file1.split('_')[1] if '_' in file1 else file1[:15]
        short2 = file2.split('_')[1] if '_' in file2 else file2[:15]
        print(f"{i:2d}. {short1} <-> {short2}: {sim:.4f}")
    
    print(f"\nTOP 10 LEAST SIMILAR PAIRS:")
    print("-" * 60)
    for i, (file1, file2, sim) in enumerate(pairs[-10:], 1):
        # Shorten filenames for display
        short1 = file1.split('_')[1] if '_' in file1 else file1[:15]
        short2 = file2.split('_')[1] if '_' in file2 else file2[:15]
        print(f"{i:2d}. {short1} <-> {short2}: {sim:.4f}")

def analyze_image_similarity(images_dir, output_dir=None, similarity_threshold=0.95):
    """
    Main function to analyze image similarity using CLIP embeddings and deduplicate similar images
    
    Args:
        images_dir: Directory containing images to analyze
        output_dir: Directory to save results (defaults to images_dir)
        similarity_threshold: Threshold for removing similar images (default: 0.95)
    """
    
    if output_dir is None:
        output_dir = images_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model...")
    embedding_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(embedding_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(embedding_model_name)
    
    # Process images and get embeddings
    embeddings, filenames = load_and_process_images(images_dir, clip_model, clip_processor, device)
    
    if embeddings is None:
        return
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings, device='cuda') # with gpu hah
    
    # Create visualizations and save results (before deduplication)
    print("\nCreating similarity analysis for original images...")
    print_similarity_stats(similarity_matrix, filenames)
    
    # Deduplicate similar images
    images_to_keep, images_to_remove = deduplicate_similar_images(
        similarity_matrix, filenames, images_dir, similarity_threshold
    )
    
    # Create deduplicated folder
    deduplicated_dir = create_deduplicated_folder(images_dir, images_to_keep, output_dir)
    
    # Update summary file with deduplication info
    summary_file = os.path.join(deduplicated_dir, "deduplication_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Image Deduplication Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Similarity threshold: {similarity_threshold}\n")
        f.write(f"Original image count: {len(filenames)}\n")
        f.write(f"Deduplicated image count: {len(images_to_keep)}\n")
        f.write(f"Images removed: {len(images_to_remove)}\n")
        f.write(f"Reduction: {len(images_to_remove)/len(filenames)*100:.1f}%\n\n")
        
        f.write("Images removed (kept the sharper one):\n")
        f.write("-" * 40 + "\n")
        for filename in sorted(images_to_remove):
            blur_score = extract_blur_score(filename)
            f.write(f"{filename} (blur score: {blur_score:.2f})\n")
        
        f.write(f"\nRemaining images:\n")
        f.write("-" * 20 + "\n")
        for filename in sorted(images_to_keep):
            blur_score = extract_blur_score(filename)
            f.write(f"{filename} (blur score: {blur_score:.2f})\n")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Final curated images in: {deduplicated_dir}")
    
    print(f"\nFinal dataset statistics:")
    print(f"  Original images: {len(filenames)}")
    print(f"  After deduplication: {len(images_to_keep)}")
    print(f"  Reduction: {len(images_to_remove)} images ({len(images_to_remove)/len(filenames)*100:.1f}%)")
    
    return deduplicated_dir

# Example usage
if __name__ == "__main__":
    # Configuration
    IMAGES_DIR = "/home/lounes/turf3/20250730_124909_processed/02_plant_detection"
    OUTPUT_DIR = "/home/lounes/turf3/20250730_124909_processed/03_similarity_analysis"
    SIMILARITY_THRESHOLD = 0.88  
    
    # Check if input directory exists
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory {IMAGES_DIR} does not exist")
        print("Make sure to run the plant detection script first")
    else:
        # Run similarity analysis with deduplication
        analyze_image_similarity(IMAGES_DIR, OUTPUT_DIR, SIMILARITY_THRESHOLD)