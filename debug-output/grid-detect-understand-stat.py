from __future__ import print_function
import cv2
import PIL.Image
import numpy as np
import sys
import os
import shutil
import itertools
import random
import glob
from time import time

import sys
sys.path.append('/workspace/sam3/seniT')

from contour_detect import *
from line_intersection import *
from rectify_refine import *

# --- Configuration ---
INPUT_DIR = '/workspace/sam3/seniT/chessboard_crops'
OUTPUT_DIR = 'debug_output'

def save_debug_step(img, step_num, description, duration=None, is_failure=False):
    """
    Saves the image with a text overlay describing the step and the time it took.
    """
    if img is None: return
    
    # Normalize float/bool images to standard 0-255 uint8
    if img.dtype == float or img.dtype == np.float64 or img.dtype == np.float32:
        save_img = (img * 255).astype(np.uint8)
    elif img.dtype == bool:
        save_img = (img.astype(np.uint8) * 255)
    else:
        save_img = img.copy()

    # Convert grayscale to color so we can draw colored text overlays
    if len(save_img.shape) == 2:
        save_img = cv2.cvtColor(save_img, cv2.COLOR_GRAY2BGR)

    # Prepare Text
    text_desc = f"{step_num:02d} {description}"
    text_time = f"{duration:.4f}s" if duration is not None else ""

    # Draw Description (Top Left) - Black border, White text
    cv2.putText(save_img, text_desc, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
    cv2.putText(save_img, text_desc, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Draw Time (Bottom Right) - Black border, Cyan text
    if text_time:
        h, w = save_img.shape[:2]
        cv2.putText(save_img, text_time, (w - 200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(save_img, text_time, (w - 200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Naming convention
    if is_failure:
        filename = f"{step_num:02d}_FAILED_ATTEMPT_{description}.png"
    else:
        filename = f"{step_num:02d}_{description}.png"
        
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, save_img)
    
    if not is_failure:
        time_msg = f" ({duration:.4f}s)" if duration is not None else ""
        print(f"Saved: {filename}{time_msg}")

def process_single_random_file(filename):
    print(f"Processing randomly selected file: {filename}")
    
    # Clean/Create output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    stats = {}
    t_total_start = time()

    # --- 1. Load and Scale ---
    t0 = time()
    img = cv2.imread(filename)
    if img is None:
        print("Failed to load image.")
        return
    # Use 1024x768 to ensure sufficient resolution for line detection
    img = scaleImageIfNeeded(img, 1024, 768)
    img_orig = img.copy()
    stats['01_Load_Scale'] = time() - t0
    save_debug_step(img, 1, "original_scaled", stats['01_Load_Scale'])

    # --- 2. Canny Edges ---
    t0 = time()
    edges = cv2.Canny(img, 100, 550)
    stats['02_Canny'] = time() - t0
    save_debug_step(edges, 2, "canny_edges", stats['02_Canny'])

    # --- 3. Morphological Gradient (Cluster detection) ---
    t0 = time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    stats['03_Morph_Grad'] = time() - t0
    save_debug_step(edges_gradient, 3, "morph_gradient", stats['03_Morph_Grad'])

    # --- 4. Find Contours ---
    t0 = time()
    contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    # Visualize raw contours
    debug_contours = np.zeros_like(img)
    cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 1)
    
    contours = simplifyContours(contours) 
    stats['04_Find_Contours'] = time() - t0
    save_debug_step(debug_contours, 4, "all_contours_raw", stats['04_Find_Contours'])
    
    # --- 5. Prune Contours ---
    t0 = time()
    contours, median_contour = pruneContours(contours)
    stats['05_Prune_Contours'] = time() - t0
    
    debug_pruned = np.zeros_like(img)
    cv2.drawContours(debug_pruned, contours, -1, (0, 255, 255), 2)
    if median_contour is not None:
        cv2.drawContours(debug_pruned, [median_contour], -1, (0, 0, 255), 4) # Red median
    save_debug_step(debug_pruned, 5, "pruned_contours", stats['05_Prune_Contours'])

    # --- 6. Mask Estimation ---
    t0 = time()
    if len(contours) > 0:
        thetas = getContourThetas(contours)
        top_two_angles = calculateKDE(thetas)
        
        # Create mask from convex hulls of contours
        mask = np.zeros(img.shape[:2], dtype=float)
        for c in contours:
            cv2.drawContours(mask, [cv2.convexHull(c)], 0, 1.0, -1)
        mask = cv2.dilate(mask, np.ones((5,5)))
    else:
        print("No contours found, using default fallback.")
        mask = np.ones(img.shape[:2], dtype=float)
        top_two_angles = [0, 90]
        min_area_rect = ((img.shape[1]/2, img.shape[0]/2), (img.shape[1], img.shape[0]), 0)
    stats['06_Mask_Est'] = time() - t0
    save_debug_step(mask, 6, "chessboard_mask_estimation", stats['06_Mask_Est'])

    # --- 7. Apply Mask ---
    t0 = time()
    edges_masked = cv2.bitwise_and(edges, edges, mask=(mask > 0.5).astype(np.uint8))
    stats['07_Apply_Mask'] = time() - t0
    save_debug_step(edges_masked, 7, "edges_masked", stats['07_Apply_Mask'])

    # --- 8. Hough Lines (Optimized & Robust) ---
    t0 = time()
    dim_min = min(img.shape[0], img.shape[1])
    lines = getHoughLines(edges_masked, min_line_size=0.25*dim_min)
    stats['08_Hough_Lines'] = time() - t0
    
    debug_lines = img.copy()
    if lines is not None:
        for line in lines:
            # FIX: Handle both (1,4) shape and flat (4,) shape
            if len(line.shape) == 2 and line.shape[0] == 1: 
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line
            cv2.line(debug_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
    save_debug_step(debug_lines, 8, "hough_lines_raw", stats['08_Hough_Lines'])

    # --- 9. Parse Lines ---
    t0 = time()
    try:
        lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=35)
    except Exception as e:
        print(f"Error parsing Hough Lines: {e}")
        return
    stats['09_Parse_Lines'] = time() - t0

    debug_parsed = img.copy()
    def draw_safe_line(img, l, color):
         pt1 = (int(l[0]), int(l[1]))
         pt2 = (int(l[2]), int(l[3]))
         cv2.line(img, pt1, pt2, color, 2)

    for l in lines_a: draw_safe_line(debug_parsed, l, (0,0,255)) # Red = Group A
    for l in lines_b: draw_safe_line(debug_parsed, l, (255,0,0)) # Blue = Group B
    save_debug_step(debug_parsed, 9, "lines_parsed_A_red_B_blue", stats['09_Parse_Lines'])

    if len(lines_a) < 2 or len(lines_b) < 2:
        print(f"Not enough structured lines found. A: {len(lines_a)}, B: {len(lines_b)}")
        return

    # --- 10. RANSAC Loop ---
    print("Running RANSAC...")
    t_ransac_start = time()
    
    best_corners = None
    final_warp_img = None
    final_M = None
    found_match = False
    failed_attempts_saved = 0
    
    # Try up to 250 combinations
    for i in range(250): 
        corners = chooseRandomGoodQuad(lines_a, lines_b, median_contour)
        if corners is None: continue

        # Calculate Transform
        M = getTileTransform(corners.astype(np.float32), tile_buffer=16, tile_res=16)
        warp_attempt, M = getTileImage(img_orig, corners.astype(np.float32), tile_buffer=16, tile_res=16)
        
        # Check for grid lines in warped image (Verification)
        lines_x, lines_y, step_x, step_y = getWarpCheckerLines(warp_attempt)
        
        # --- Visualize RANSAC Failures (First 3 only) ---
        if len(lines_x) == 0 and failed_attempts_saved < 3:
            debug_fail = img.copy()
            pts = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(debug_fail, [pts], True, (0, 0, 255), 3) # Red box
            save_debug_step(debug_fail, 10, f"ransac_fail_quad_{i}", is_failure=True)
            failed_attempts_saved += 1
            
        if len(lines_x) > 0:
            print(f"RANSAC Match found at iteration {i}")
            best_corners = corners
            final_warp_img = warp_attempt
            final_M = M
            found_match = True
            break
            
    stats['10_RANSAC_Total'] = time() - t_ransac_start

    if not found_match:
        print("RANSAC failed to find a valid grid.")
        return

    # Visualize Winning Quad
    debug_quad = img.copy()
    pts = best_corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(debug_quad, [pts], True, (0, 255, 0), 3) # Green box
    save_debug_step(debug_quad, 10, "ransac_winning_quad", stats['10_RANSAC_Total'])
    save_debug_step(final_warp_img, 11, "ransac_initial_warp") 

    # --- 11. Refinement ---
    print("Refining...")
    t_refine_start = time()
    
    # 11a. Visualize Detected Grid Lines in Warp
    debug_warp_lines = final_warp_img.copy()
    if len(debug_warp_lines.shape) == 2: 
        debug_warp_lines = cv2.cvtColor(debug_warp_lines, cv2.COLOR_GRAY2BGR)
    for x_val in lines_x:
        cv2.line(debug_warp_lines, (int(x_val), 0), (int(x_val), debug_warp_lines.shape[0]), (0, 255, 0), 1)
    for y_val in lines_y:
        cv2.line(debug_warp_lines, (0, int(y_val)), (debug_warp_lines.shape[1], int(y_val)), (0, 0, 255), 1)
    save_debug_step(debug_warp_lines, 12, "refine_detected_grid_lines")

    # 11b. Calculate Rectilinear Corners
    warp_corners, all_warp_corners = getRectChessCorners(lines_x, lines_y)
    debug_warp_corners = debug_warp_lines.copy()
    for c in warp_corners:
        cv2.circle(debug_warp_corners, tuple(map(int, c)), 4, (255, 255, 0), -1) # Teal dots
    save_debug_step(debug_warp_corners, 13, "refine_corners_calculated")

    # 11c. Map Back to Original Image
    tile_centers = all_warp_corners + np.array([step_x/2.0, step_y/2.0])
    M_inv = np.matrix(np.linalg.inv(final_M))
    real_corners, all_real_tile_centers = getOrigChessCorners(warp_corners, tile_centers, M_inv)
    
    debug_refined_1 = img_orig.copy()
    cv2.polylines(debug_refined_1, [real_corners.astype(np.int32)], True, (255, 0, 0), 2)
    save_debug_step(debug_refined_1, 14, "refine_mapped_back_to_orig")

    # 11d. Second Warp (Higher precision)
    tile_res = 64
    tile_buffer = 1
    warp_img_2, better_M = getTileImage(img_orig, real_corners, tile_buffer=tile_buffer, tile_res=tile_res)
    save_debug_step(warp_img_2, 15, "refine_second_warp_cleaner")

    # 11e. Re-Rectify (Fix Rotation)
    warp_img_final, was_rotated, refine_M = reRectifyImages(warp_img_2)
    save_debug_step(warp_img_final, 16, "refine_final_rectified")

    # 11f. Final Coordinate Calculation
    combined_M = np.matmul(refine_M, better_M)
    M_inv_final = np.matrix(np.linalg.inv(combined_M))

    hlines = vlines = (np.arange(8)+tile_buffer)*tile_res
    hcorner = (np.array([0,8,8,0])+tile_buffer)*tile_res
    vcorner = (np.array([0,0,8,8])+tile_buffer)*tile_res
    ideal_corners = np.vstack([hcorner,vcorner]).T
    
    final_real_corners, _ = getOrigChessCorners(ideal_corners, ideal_corners, M_inv_final)
    
    stats['11_Refinement_Total'] = time() - t_refine_start

    # Final Result Overlay
    debug_final = img_orig.copy()
    cv2.polylines(debug_final, [final_real_corners.astype(np.int32)], True, (0, 255, 0), 3)
    save_debug_step(debug_final, 17, "FINAL_RESULT_overlay", stats['11_Refinement_Total'])
    
    total_time = time() - t_total_start
    
    # --- Print Stats ---
    print("\n" + "="*40)
    print(f"TIMING REPORT (Total: {total_time:.4f}s)")
    print("-" * 40)
    for k, v in sorted(stats.items()):
        print(f"{k:<25} : {v:.4f} s")
    print("="*40 + "\n")
    print("Processing complete. Check 'debug_output' folder.")

if __name__ == '__main__':
    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    filenames = []
    for ext in extensions:
        filenames.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not filenames:
        print(f"No images found in {INPUT_DIR}")
        sys.exit(1)
        
    # Select ONE random file
    target_file = random.choice(filenames)
    process_single_random_file(target_file)