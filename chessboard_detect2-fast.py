

# THE IDEA IS TO REFINE ONLY IF THE GRID IS NOT ON INSIDE THE BBOX
# ----> it is not working but will get to it later 


from __future__ import print_function
import cv2
import PIL.Image
import numpy as np
import sys
import os
import shutil
import itertools
from time import time
from matplotlib import pyplot as plt

# Assuming these imports exist in your environment
from contour_detect import  *
from line_intersection import  *
from rectify_refine import *

np.set_printoptions(suppress=True, precision=2, linewidth=200)

def getChessboardInfoFromFullImage(img, edges):
    # Morphological Gradient to get internal squares of canny edges. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = simplifyContours(contours)
    contours, median_contour = pruneContours(contours)

    if len(contours) == 0:
       return np.ones(img.shape[:2], dtype=float), [0, 90], ((img.shape[1]/2, img.shape[0]/2), (img.shape[1], img.shape[0]), 0), None

    thetas = getContourThetas(contours)
    top_two_angles = calculateKDE(thetas)

    mask = np.ones(img.shape[:2], dtype=float)
    min_area_rect = ((img.shape[1]/2, img.shape[0]/2), (img.shape[1], img.shape[0]), 0)

    return mask, top_two_angles, min_area_rect, median_contour

def processFile(filename):
    stats = {}
    t_start = time()
    
    img = cv2.imread(filename)
    # img = scaleImageIfNeeded(img, 1024, 768)
    img = scaleImageIfNeeded(img, 1024, 768)
    img_orig = img.copy()
    img_orig2 = img.copy()
    
    stats['1_load_scale'] = time() - t_start
    t_last = time()

    # Edges
    edges = cv2.Canny(img, 100, 550)
    
    stats['2_canny'] = time() - t_last
    t_last = time()

    # Mask
    mask, top_two_angles, min_area_rect, median_contour = getChessboardInfoFromFullImage(img, edges)
    
    stats['3_mask_estimation'] = time() - t_last
    t_last = time()

    # Hough Lines
    edges_masked = cv2.bitwise_and(edges,edges,mask = (mask > 0.5).astype(np.uint8))
    img_orig = cv2.bitwise_and(img_orig,img_orig,mask = (mask > 0.5).astype(np.uint8))

    lines = getHoughLines(edges_masked, min_line_size=0.25*min(min_area_rect[1]))
    
    stats['4_hough_lines'] = time() - t_last
    t_last = time()

    lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=35)
    
    stats['5_parse_lines'] = time() - t_last
    t_last = time()
    
    if len(lines_a) < 2 or len(lines_b) < 2:
        stats['6_ransac'] = 0
        stats['7_refinement'] = 0
        return img_orig, edges_masked, img_orig, stats, False, f"Insufficient structured lines (A: {len(lines_a)}, B: {len(lines_b)})"

    # --- RANSAC START ---
    t_ransac_start = time()
    total_outer_loops = 0
    total_inner_loops = 0
    t_accum_quad_select = 0
    t_accum_transform = 0
    t_accum_angle_check = 0
    t_accum_verification = 0

    lines_x, lines_y, step_x, step_y = [], [], 0, 0
    corners = np.array([])
    M = np.eye(3)

    for i2 in range(5):
        total_outer_loops += 1
        for i in range(50):
            total_inner_loops += 1
            
            # 1. Quad Selection
            t0 = time()
            corners = chooseRandomGoodQuad(lines_a, lines_b, median_contour)
            t_accum_quad_select += time() - t0
            
            # 2. Transform Calculation
            t0 = time()
            M = getTileTransform(corners.astype(np.float32),tile_buffer=16, tile_res=16)

            all_lines = np.vstack([lines_a[:,:2], lines_a[:,2:], lines_b[:,:2], lines_b[:,2:]]).astype(np.float32)
            warp_pts = cv2.perspectiveTransform(all_lines[None,:,:], M)
            warp_pts = warp_pts[0,:,:]
            warp_lines_a = np.hstack([warp_pts[:len(lines_a),:], warp_pts[len(lines_a):2*len(lines_a),:]])
            warp_lines_b = np.hstack([warp_pts[2*len(lines_a):2*len(lines_a)+len(lines_b),:], warp_pts[2*len(lines_a)+len(lines_b):,:]])
            t_accum_transform += time() - t0

            # 3. Angle Checks
            t0 = time()
            thetas_a = np.array([getSegmentTheta(line) for line in warp_lines_a])
            thetas_b = np.array([getSegmentTheta(line) for line in warp_lines_b])
            median_theta_a = (np.median(thetas_a*180/np.pi))
            median_theta_b = (np.median(thetas_b*180/np.pi))
            
            if i < 20: warp_angle_threshold = 0.03
            elif i < 30: warp_angle_threshold = 0.1
            elif i < 50: warp_angle_threshold = 0.3
            elif i < 70: warp_angle_threshold = 0.5
            elif i < 80: warp_angle_threshold = 1.0
            else: warp_angle_threshold = 2.0
            
            match_found = False
            if ((angleCloseDeg(abs(median_theta_a), 0, warp_angle_threshold) and 
                 angleCloseDeg(abs(median_theta_b), 90, warp_angle_threshold)) or 
                (angleCloseDeg(abs(median_theta_a), 90, warp_angle_threshold) and 
                 angleCloseDeg(abs(median_theta_b), 0, warp_angle_threshold))):
                match_found = True
            
            t_accum_angle_check += time() - t0
            
            if match_found:
                break

        # 4. Verification
        t0 = time()
        warp_img, M = getTileImage(img_orig, corners.astype(np.float32),tile_buffer=16, tile_res=16)
        lines_x, lines_y, step_x, step_y = getWarpCheckerLines(warp_img)
        t_accum_verification += time() - t0

        if len(lines_x) > 0:
            print('Found good chess lines (%d): %s %s' % (i2, lines_x, lines_y))
            break
            
    print("Ransac detection took %.4f seconds." % (time() - t_ransac_start))
    
    stats['6_ransac'] = time() - t_last
    stats['6_ransac_outer_loops'] = total_outer_loops
    stats['6_step_4_verify_img'] = t_accum_verification
    t_last = time()
    # --- RANSAC END ---

    warp_img, M = getTileImage(img_orig, corners.astype(np.float32),tile_buffer=16, tile_res=16)

    for corner in corners:
        cv2.circle(img, tuple(map(int,corner)), 5, (255,150,150),-1)  

    if len(lines_x) > 0:
        t_refine_start = time()
        
        # 7a. Initial Corner Calculation 
        warp_corners, all_warp_corners = getRectChessCorners(lines_x, lines_y)
        tile_centers = all_warp_corners + np.array([step_x/2.0, step_y/2.0])
        M_inv = np.matrix(np.linalg.inv(M))
        real_corners, all_real_tile_centers = getOrigChessCorners(warp_corners, tile_centers, M_inv)
        
        # --- FIX: Convert matrix to array to avoid "length-1" type errors ---
        real_corners = np.asarray(real_corners)
        all_real_tile_centers = np.asarray(all_real_tile_centers)
        
        stats['7a_calc_corners'] = time() - t_refine_start
        
        # --- NEW FILTER: Check Mask/Bounds Intersection ---
        needs_refinement = False
        img_h, img_w = img.shape[:2]
        
        for corner in real_corners:
            cx, cy = int(corner[0]), int(corner[1])
            
            # Check 1: Corner is outside image bounds
            if cx < 0 or cx >= img_w or cy < 0 or cy >= img_h:
                needs_refinement = True
                break
            
            # Check 2: Corner is outside the detected chessboard mask
            if mask[cy, cx] < 0.5:
                needs_refinement = True
                break
        
        stats['7_needs_refinement'] = 1 if needs_refinement else 0

        if needs_refinement:
            print("Corner out of bounds/mask. Refining...")
            t_refine_loop = time()
            
            tile_res = 64 
            tile_buffer = 1
            
            # 7b. First Warp (Refinement)
            warp_img, better_M = getTileImage(img_orig2, real_corners, tile_buffer=tile_buffer, tile_res=tile_res)
            
            # 7d. Final Corner Calculation based on ideal grid
            combined_M = better_M
            M_inv = np.matrix(np.linalg.inv(combined_M))

            hlines = vlines = (np.arange(8)+tile_buffer)*tile_res
            hcorner = (np.array([0,8,8,0])+tile_buffer)*tile_res
            vcorner = (np.array([0,0,8,8])+tile_buffer)*tile_res
            ideal_corners = np.vstack([hcorner,vcorner]).T
            ideal_all_corners = np.array(list(itertools.product(hlines, vlines)))
            ideal_tile_centers = ideal_all_corners + np.array([tile_res/2.0, tile_res/2.0]) 

            real_corners, all_real_tile_centers = getOrigChessCorners(ideal_corners, ideal_tile_centers, M_inv)
            
            # Ensure these are arrays too for subsequent drawing
            real_corners = np.asarray(real_corners)
            all_real_tile_centers = np.asarray(all_real_tile_centers)
            
            stats['7b_refine_loop'] = time() - t_refine_loop
            
            # Update warp image for return
            warp_img, _ = getTileImage(img_orig2, real_corners, tile_buffer=tile_buffer, tile_res=tile_res)
        else:
            print("Grid fits inside mask. Skipping refinement.")
            tile_res = 64
            tile_buffer = 1
            warp_img, _ = getTileImage(img_orig2, real_corners, tile_buffer=tile_buffer, tile_res=tile_res)

        # 7e. Draw (Common)
        cv2.polylines(img, [real_corners.astype(np.int32)], True, (150,50,255), thickness=3)
        cv2.polylines(img, [all_real_tile_centers.astype(np.int32)], False, (0,50,255), thickness=1)
        cv2.drawContours(mask,[real_corners.astype(int)],0,1,-1)
        
    stats['7_total_refinement_time'] = time() - t_last

    img_masked_full = cv2.bitwise_and(img,img,mask = (mask > 0.5).astype(np.uint8))
    img_masked = cv2.addWeighted(img,0.2,img_masked_full,0.8,0)

    drawMinAreaRect(img_masked, min_area_rect)

    success = len(lines_x) > 0
    msg = "Success" if success else "RANSAC failed to find grid pattern"

    return img_masked, edges_masked, warp_img, stats, success, msg

def main(filenames):
    processing_times = []
    step_stats = {}
    
    success_times = []
    failure_times = []
    failure_reasons = {}
    
    refined_count = 0
    skipped_count = 0
    
    log_file = open("processing_failures.log", "w")
    
    failed_dir = "failed_images"
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir)
    
    for filename in filenames:
        a = time()
        success = False
        reason = "Exception"
        
        try:
            img_masked, edges_masked, warp_img, stats, success, reason = processFile(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            reason = f"Exception: {str(e)}"
            duration = time() - a
            failure_times.append(duration)
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            log_file.write(f"{filename}: {reason} (Time: {duration:.4f}s)\n")
            continue
            
        duration = time() - a
        processing_times.append(duration)
        
        if success:
            success_times.append(duration)
            out_filename = filename[:-4].replace('/','_').replace('\\','_')
            PIL.Image.fromarray(cv2.cvtColor(img_masked,cv2.COLOR_BGR2RGB)).save("rectified2samMask/%s_overlay.png" % out_filename)
            
            if stats.get('7_needs_refinement', 0) == 1:
                refined_count += 1
            else:
                skipped_count += 1
        else:
            failure_times.append(duration)
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            log_file.write(f"{filename}: {reason} (Time: {duration:.4f}s)\n")
            out_filename = filename[:-4].replace('/','_').replace('\\','_')
            PIL.Image.fromarray(cv2.cvtColor(img_masked,cv2.COLOR_BGR2RGB)).save("rectified2samMask/%s_overlay_FAILED.png" % out_filename)
            
            try:
                shutil.copy(filename, os.path.join(failed_dir, os.path.basename(filename)))
            except Exception as e:
                print(f"Error copying failed file {filename}: {e}")
        
        for k, v in stats.items():
            if k not in step_stats:
                step_stats[k] = []
            step_stats[k].append(v)
            
        print(f"Processed {filename}: {reason} ({duration:.4f}s)")

    log_file.close()

    if processing_times:
        print("\n" + "="*40)
        print("Processing Statistics:")
        print(f"Total images: {len(processing_times)}")
        print(f"Success: {len(success_times)} ({len(success_times)/len(processing_times)*100:.1f}%)")
        print(f"  - Refined: {refined_count}")
        print(f"  - Skipped Refinement: {skipped_count}")
        print(f"Failed: {len(failure_times)} ({len(failure_times)/len(processing_times)*100:.1f}%)")
        
        if success_times:
            print(f"Avg Success Time: {sum(success_times)/len(success_times):.4f} s")
            
        print("\nStep-wise Statistics:")
        for k in sorted(step_stats.keys()):
            values = np.array(step_stats[k])
            avg_step = np.mean(values)
            if 'loops' in k or 'needs_refinement' in k:
                 print(f"  {k}: {avg_step:.2f}")
            else:
                 print(f"  {k}: {avg_step:.4f} s")
        
        print("="*40 + "\n")
    else:
        print("No images were successfully processed.")

if __name__ == '__main__':
    import glob
    
    input_dir = '/workspace/sam3/seniT/chessboard_crops'
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    filenames = []
    for ext in extensions:
        filenames.extend(glob.glob(os.path.join(input_dir, ext)))
    filenames.sort()
    
    if not filenames:
        print(f"No images found in {input_dir}")
        sys.exit(1)
        
    if not os.path.exists("rectified2samMask"):
        os.makedirs("rectified2samMask")
        
    main(filenames)