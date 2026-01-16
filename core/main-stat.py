import os
import glob
import cv2
import numpy as np
import time
import argparse
import sys
from src.segmentation import ChessboardSegmenter
from src.grid_solver import GridSolver
from src.filter import QualityFilter

TOTAL_TIME_LIMIT_MS = 300.0 

def main(input_dir, output_dir, assets_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"Initializing Engines from {assets_dir}...")
    try:
        segmenter = ChessboardSegmenter(assets_dir)
        solver = GridSolver()
        quality_filter = QualityFilter(min_score=0.4)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
                         glob.glob(os.path.join(input_dir, '*.png')))
    print(f"Found {len(image_files)} images.")

    print("Warming up GPU...")
    segmenter.predict('warmup.jpg') if os.path.exists('warmup.jpg') else None
    print("Warmup done.\n")

    stats_total = []
    stats_skipped = 0
    
    print(f"{'FILENAME':<25} | {'SEG (ms)':<10} | {'GRID (ms)':<10} | {'TOTAL':<10} | {'STATUS'}")
    print("-" * 80)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        t_start = time.perf_counter()
        
        try:
            # 1. Segmentation
            mask, original_img, score = segmenter.predict(img_path)
            
            t_seg_end = time.perf_counter()
            seg_ms = (t_seg_end - t_start) * 1000.0

            if seg_ms > TOTAL_TIME_LIMIT_MS:
                print(f"{filename:<25} | {seg_ms:<10.2f} | {'-':<10} | {seg_ms:<10.2f} | SKIP (Seg Slow)")
                stats_skipped += 1
                continue

            # 2. Quality Filter
            is_good, reason = quality_filter.check(mask, score)
            if not is_good:
                print(f"{filename:<25} | {seg_ms:<10.2f} | {'-':<10} | {seg_ms:<10.2f} | SKIP ({reason})")
                stats_skipped += 1
                continue

            # --- 3. FOCUS STEP: Mask & Crop ---
            # A. Get Bounding Box
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # B. Black out background (Keep only chessboard)
            # bitwise_and needs src1 and src2 to be same size, mask needs to be single channel
            masked_full_img = cv2.bitwise_and(original_img, original_img, mask=mask)
            
            # C. Crop to Bounding Box
            # We add a tiny margin (e.g. 5px) to ensure we don't cut the edge line, 
            # but usually exact bbox is fine if segmentation is good.
            # Let's use the exact bbox to keep it strictly "focused".
            crop_img = masked_full_img[y:y+h, x:x+w]
            crop_mask = mask[y:y+h, x:x+w]

            # 4. Grid Solving (On the Crop)
            remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
            
            # Pass the CROP to the solver
            warped_grid, corners_local, _, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=remaining_ms)
            
            t_grid_end = time.perf_counter()
            grid_ms = (t_grid_end - t_seg_end) * 1000.0
            total_ms = (t_grid_end - t_start) * 1000.0

            if 'timeout_at' in stats:
                print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | SKIP (Timeout: {stats['timeout_at']})")
                stats_skipped += 1
                continue
            
            if not success:
                print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | FAILED ({stats.get('error','')})")
                stats_skipped += 1
                continue

            # 5. Coordinate Mapping (Local -> Global)
            # The solver found corners relative to `crop_img`. 
            # We must add (x, y) to map them back to `original_img`.
            corners_global = corners_local + np.array([x, y])

            # 6. Save Results
            vis_img = original_img.copy()
            cv2.polylines(vis_img, [corners_global.astype(np.int32)], True, (0, 255, 0), 3)
            
            cv2.imwrite(os.path.join(output_dir, filename), vis_img)
            # Optionally save the cropped view for debug
            # cv2.imwrite(os.path.join(output_dir, f"crop_{filename}"), crop_img)
            
            stats_total.append(total_ms)
            print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | OK")

        except Exception as e:
            print(f"{filename:<25} | ERROR: {str(e)}")
            stats_skipped += 1

    print("-" * 80)
    print(f"Processed (OK): {len(stats_total)}")
    print(f"Skipped / Failed: {stats_skipped}")
    if stats_total:
        print(f"Average Latency (OK): {sum(stats_total)/len(stats_total):.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../vid1", help="Input directory")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--assets", default="./assets", help="Assets directory")
    args = parser.parse_args()
    
    main(args.input, args.output, args.assets)


# import os
# import glob
# import cv2
# import numpy as np
# import time
# import argparse
# import sys
# from src.segmentation import ChessboardSegmenter
# from src.grid_solver import GridSolver

# # HARD LIMIT for Total Frame Processing
# TOTAL_TIME_LIMIT_MS = 300.0 

# def main(input_dir, output_dir, assets_dir):
#     # Setup
#     if not os.path.exists(output_dir): os.makedirs(output_dir)

#     print(f"Initializing Engines from {assets_dir}...")
#     try:
#         segmenter = ChessboardSegmenter(assets_dir)
#         solver = GridSolver()
#     except Exception as e:
#         print(f"CRITICAL ERROR: {e}")
#         sys.exit(1)

#     image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
#                          glob.glob(os.path.join(input_dir, '*.png')))
#     print(f"Found {len(image_files)} images.")

#     # Warmup
#     print("Warming up GPU...")
#     segmenter.predict('warmup.jpg') if os.path.exists('warmup.jpg') else None
#     print("Warmup done.\n")

#     stats_total = []
#     stats_skipped = 0
    
#     print(f"{'FILENAME':<25} | {'SEG (ms)':<10} | {'GRID (ms)':<10} | {'TOTAL':<10} | {'STATUS'}")
#     print("-" * 80)

#     for img_path in image_files:
#         filename = os.path.basename(img_path)
        
#         # 1. Start Global Timer
#         t_start = time.perf_counter()
        
#         try:
#             # --- Step A: Segmentation ---
#             mask, original_img = segmenter.predict(img_path)
            
#             t_seg_end = time.perf_counter()
#             seg_ms = (t_seg_end - t_start) * 1000.0

#             # CHECK 1: Global Timeout already exceeded?
#             if seg_ms > TOTAL_TIME_LIMIT_MS:
#                 print(f"{filename:<25} | {seg_ms:<10.2f} | {'-':<10} | {seg_ms:<10.2f} | SKIP (Seg Slow)")
#                 stats_skipped += 1
#                 continue

#             # --- Step B: Calculate Remaining Budget ---
#             # The grid solver must finish within this time to stay under 300ms total
#             remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
            
#             # --- Step C: Grid Solving (With Budget) ---
#             warped_grid, corners, success, stats = solver.solve(original_img, mask, time_limit_ms=remaining_ms)
            
#             t_grid_end = time.perf_counter()
#             grid_ms = (t_grid_end - t_seg_end) * 1000.0
#             total_ms = (t_grid_end - t_start) * 1000.0

#             # CHECK 2: Did GridSolver return a timeout?
#             if 'timeout_at' in stats:
#                 print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | SKIP (Timeout: {stats['timeout_at']})")
#                 stats_skipped += 1
#                 continue
            
#             # CHECK 3: Did it fail algorithmically?
#             if not success:
#                 print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | FAILED ({stats.get('error','')})")
#                 stats_skipped += 1 # Count failure as skipped/failed
#                 continue

#             # --- Success ---
#             # Save results...
#             vis_img = original_img.copy()
#             cv2.polylines(vis_img, [corners.astype(np.int32)], True, (0, 255, 0), 3)
#             cv2.imwrite(os.path.join(output_dir, filename), vis_img)
            
#             stats_total.append(total_ms)
#             print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | OK")

#         except Exception as e:
#             print(f"{filename:<25} | ERROR: {str(e)}")
#             stats_skipped += 1

#     # Summary
#     print("-" * 80)
#     print(f"Processed (OK): {len(stats_total)}")
#     print(f"Skipped / Failed: {stats_skipped}")
#     if stats_total:
#         print(f"Average Latency (OK): {sum(stats_total)/len(stats_total):.2f} ms")
        

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", default="./data", help="Input directory")
#     parser.add_argument("--output", default="./output", help="Output directory")
#     parser.add_argument("--assets", default="./assets", help="Assets directory")
#     args = parser.parse_args()
    
#     main(args.input, args.output, args.assets)