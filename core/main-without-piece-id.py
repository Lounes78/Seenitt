import os
import glob
import cv2
import numpy as np
import time
import sys
import argparse
from src.segmentation import ChessboardSegmenter
from src.grid_solver import GridSolver
from src.filter import QualityFilter

TOTAL_TIME_LIMIT_MS = 300.0 

def main(input_dir, output_dir, assets_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Logging setup
    log_path = os.path.join(output_dir, "filtered_frames.log")
    log_file = open(log_path, "w")
    print(f"Logging filtered frames to: {log_path}")

    print(f"Initializing Engines from {assets_dir}...")
    try:
        segmenter = ChessboardSegmenter(assets_dir)
        solver = GridSolver()
        # Initialize Logic: Strict Margin Check
        quality_filter = QualityFilter(margin=1, min_score=0.5, max_border_contact_ratio=0.05)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
                         glob.glob(os.path.join(input_dir, '*.png')))
    print(f"Found {len(image_files)} images.")

    print("Warming up GPU...")
    if os.path.exists('warmup.jpg'):
        segmenter.predict('warmup.jpg')
    else:
        dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
        cv2.imwrite('warmup_dummy.jpg', dummy)
        segmenter.predict('warmup_dummy.jpg')
        os.remove('warmup_dummy.jpg')
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
                msg = "SKIP (Seg Slow)"
                print(f"{filename:<25} | {seg_ms:<10.2f} | {'-':<10} | {seg_ms:<10.2f} | {msg}")
                log_file.write(f"{filename}: {msg}\n")
                stats_skipped += 1
                continue

            # 2. Quality Filter (Rigid Margin Check)
            is_good, reason = quality_filter.check(mask, score)
            if not is_good:
                msg = f"SKIP ({reason})"
                print(f"{filename:<25} | {seg_ms:<10.2f} | {'-':<10} | {seg_ms:<10.2f} | {msg}")
                log_file.write(f"{filename}: {msg}\n")
                stats_skipped += 1
                continue

            # --- 3. FOCUS STEP: Mask & Crop ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Mask out background
            masked_full_img = cv2.bitwise_and(original_img, original_img, mask=mask)
            
            # Crop to detected board
            crop_img = masked_full_img[y:y+h, x:x+w]
            crop_mask = mask[y:y+h, x:x+w]

            # 4. Grid Solving (On the Crop)
            remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
            
            warped_grid, corners_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=remaining_ms)
            
            t_grid_end = time.perf_counter()
            grid_ms = (t_grid_end - t_seg_end) * 1000.0
            total_ms = (t_grid_end - t_start) * 1000.0

            if 'timeout_at' in stats:
                msg = f"SKIP (Timeout: {stats['timeout_at']})"
                print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | {msg}")
                log_file.write(f"{filename}: {msg}\n")
                stats_skipped += 1
                continue
            
            if not success:
                msg = f"FAILED ({stats.get('error','')})"
                print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | {msg}")
                log_file.write(f"{filename}: {msg}\n")
                stats_skipped += 1
                continue

            # 5. Success
            corners_global = corners_local + np.array([x, y])

            vis_img = original_img.copy()
            cv2.polylines(vis_img, [corners_global.astype(np.int32)], True, (0, 255, 0), 3)
            
            cv2.imwrite(os.path.join(output_dir, filename), vis_img)
            
            stats_total.append(total_ms)
            print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | OK")

        except Exception as e:
            msg = f"ERROR: {str(e)}"
            print(f"{filename:<25} | {msg}")
            log_file.write(f"{filename}: {msg}\n")
            stats_skipped += 1

    log_file.close()
    
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
# import sys
# import argparse
# from src.segmentation import ChessboardSegmenter
# from src.grid_solver import GridSolver
# from src.filter import QualityFilter

# TOTAL_TIME_LIMIT_MS = 300.0 

# def main(input_dir, output_dir, assets_dir):
#     # Setup Output
#     if not os.path.exists(output_dir): os.makedirs(output_dir)
    
#     # Setup Logging
#     log_path = os.path.join(output_dir, "filtered_frames.log")
#     log_file = open(log_path, "w")
#     print(f"Logging filtered frames to: {log_path}")

#     print(f"Initializing Engines from {assets_dir}...")
#     try:
#         segmenter = ChessboardSegmenter(assets_dir)
#         solver = GridSolver()
#         # Initialize with new logic (Default: min_score=0.5, min_area_ratio=0.05)
#         quality_filter = QualityFilter(min_score=0.5, min_area_ratio=0.05)
#     except Exception as e:
#         print(f"CRITICAL ERROR: {e}")
#         sys.exit(1)

#     image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
#                          glob.glob(os.path.join(input_dir, '*.png')))
#     print(f"Found {len(image_files)} images.")

#     print("Warming up GPU...")
#     # Basic warmup attempt
#     if os.path.exists('warmup.jpg'):
#         segmenter.predict('warmup.jpg')
#     else:
#         # Create dummy image for warmup if file doesn't exist
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         cv2.imwrite('warmup_dummy.jpg', dummy)
#         segmenter.predict('warmup_dummy.jpg')
#         os.remove('warmup_dummy.jpg')
#     print("Warmup done.\n")

#     stats_total = []
#     stats_skipped = 0
    
#     print(f"{'FILENAME':<25} | {'SEG (ms)':<10} | {'GRID (ms)':<10} | {'TOTAL':<10} | {'STATUS'}")
#     print("-" * 80)

#     for img_path in image_files:
#         filename = os.path.basename(img_path)
#         t_start = time.perf_counter()
        
#         try:
#             # 1. Segmentation
#             mask, original_img, score = segmenter.predict(img_path)
            
#             t_seg_end = time.perf_counter()
#             seg_ms = (t_seg_end - t_start) * 1000.0

#             if seg_ms > TOTAL_TIME_LIMIT_MS:
#                 msg = "SKIP (Seg Slow)"
#                 print(f"{filename:<25} | {seg_ms:<10.2f} | {'-':<10} | {seg_ms:<10.2f} | {msg}")
#                 log_file.write(f"{filename}: {msg}\n")
#                 stats_skipped += 1
#                 continue

#             # 2. Quality Filter (New Quad Logic)
#             is_good, reason = quality_filter.check(mask, score)
#             if not is_good:
#                 msg = f"SKIP ({reason})"
#                 print(f"{filename:<25} | {seg_ms:<10.2f} | {'-':<10} | {seg_ms:<10.2f} | {msg}")
#                 log_file.write(f"{filename}: {msg}\n")
#                 stats_skipped += 1
#                 continue

#             # --- 3. FOCUS STEP: Mask & Crop ---
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
            
#             masked_full_img = cv2.bitwise_and(original_img, original_img, mask=mask)
            
#             crop_img = masked_full_img[y:y+h, x:x+w]
#             crop_mask = mask[y:y+h, x:x+w]

#             # 4. Grid Solving (On the Crop)
#             remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
            
#             warped_grid, corners_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=remaining_ms)
            
#             t_grid_end = time.perf_counter()
#             grid_ms = (t_grid_end - t_seg_end) * 1000.0
#             total_ms = (t_grid_end - t_start) * 1000.0

#             if 'timeout_at' in stats:
#                 msg = f"SKIP (Timeout: {stats['timeout_at']})"
#                 print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | {msg}")
#                 log_file.write(f"{filename}: {msg}\n")
#                 stats_skipped += 1
#                 continue
            
#             if not success:
#                 msg = f"FAILED ({stats.get('error','')})"
#                 print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | {msg}")
#                 log_file.write(f"{filename}: {msg}\n")
#                 stats_skipped += 1
#                 continue

#             # 5. Success: Map Coordinates & Save
#             corners_global = corners_local + np.array([x, y])

#             vis_img = original_img.copy()
#             cv2.polylines(vis_img, [corners_global.astype(np.int32)], True, (0, 255, 0), 3)
            
#             cv2.imwrite(os.path.join(output_dir, filename), vis_img)
            
#             stats_total.append(total_ms)
#             print(f"{filename:<25} | {seg_ms:<10.2f} | {grid_ms:<10.2f} | {total_ms:<10.2f} | OK")

#         except Exception as e:
#             msg = f"ERROR: {str(e)}"
#             print(f"{filename:<25} | {msg}")
#             log_file.write(f"{filename}: {msg}\n")
#             stats_skipped += 1

#     log_file.close()
    
#     print("-" * 80)
#     print(f"Processed (OK): {len(stats_total)}")
#     print(f"Skipped / Failed: {stats_skipped}")
#     if stats_total:
#         print(f"Average Latency (OK): {sum(stats_total)/len(stats_total):.2f} ms")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", default="../vid1", help="Input directory")
#     parser.add_argument("--output", default="./output", help="Output directory")
#     parser.add_argument("--assets", default="./assets", help="Assets directory")
#     args = parser.parse_args()
    
#     main(args.input, args.output, args.assets)