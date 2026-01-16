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
    
    log_path = os.path.join(output_dir, "filtered_frames.log")
    log_file = open(log_path, "w")
    print(f"Logging filtered frames to: {log_path}")

    print(f"Initializing Engines from {assets_dir}...")
    try:
        segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
        solver = GridSolver()
        quality_filter = QualityFilter(margin=1, min_score=0.5, max_border_contact_ratio=0.05)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
                         glob.glob(os.path.join(input_dir, '*.png')))
    print(f"Found {len(image_files)} images.")

    print("Warming up GPU...")
    if os.path.exists('warmup.jpg'):
        segmenter.predict('warmup.jpg', "chessboard")
    else:
        dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
        cv2.imwrite('warmup_dummy.jpg', dummy)
        segmenter.predict('warmup_dummy.jpg', "chessboard")
        os.remove('warmup_dummy.jpg')
    print("Warmup done.\n")

    stats_total = []
    stats_skipped = 0
    
    print(f"{'FILENAME':<25} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS'}")
    print("-" * 95)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        t_start = time.perf_counter()
        
        try:
            # 1. Segmentation (Board)
            mask, original_img, score = segmenter.predict(img_path, "chessboard")
            t_seg_end = time.perf_counter()
            seg_ms = (t_seg_end - t_start) * 1000.0

            if seg_ms > TOTAL_TIME_LIMIT_MS:
                msg = "SKIP (Seg Slow)"
                print(f"{filename:<25} | {seg_ms:<8.1f} | {'-':<8} | {'-':<8} | {seg_ms:<8.1f} | {msg}")
                log_file.write(f"{filename}: {msg}\n")
                stats_skipped += 1
                continue

            # 2. Filter
            is_good, reason = quality_filter.check(mask, score)
            if not is_good:
                msg = f"SKIP ({reason})"
                print(f"{filename:<25} | {seg_ms:<8.1f} | {'-':<8} | {'-':<8} | {seg_ms:<8.1f} | {msg}")
                log_file.write(f"{filename}: {msg}\n")
                stats_skipped += 1
                continue

            # 3. Focus Step (Crop)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Use original image for crop (don't mask background to black)
            # This prevents cutting off pieces that stick out of the board mask
            crop_img = original_img[y:y+h, x:x+w].copy()
            crop_mask = mask[y:y+h, x:x+w]

            # 4. Grid Solving
            remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
            warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=remaining_ms)
            
            t_grid_end = time.perf_counter()
            grid_ms = (t_grid_end - t_seg_end) * 1000.0

            if not success or 'timeout_at' in stats:
                status = f"SKIP (Timeout: {stats['timeout_at']})" if 'timeout_at' in stats else f"FAILED ({stats.get('error','')})"
                total_ms = (t_grid_end - t_start) * 1000.0
                print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {'-':<8} | {total_ms:<8.1f} | {status}")
                log_file.write(f"{filename}: {status}\n")
                stats_skipped += 1
                continue

            # 5. Piece Detection
            # Passing the unmasked crop ensures full pieces are seen
            piece_masks_list, _, _ = segmenter.predict(crop_img, "chess pieces")
            t_pieces_end = time.perf_counter()
            pieces_ms = (t_pieces_end - t_grid_end) * 1000.0
            total_ms = (t_pieces_end - t_start) * 1000.0

            # 6. Visualization & Mapping
            vis_img = original_img.copy()
            grid_points_global = grid_points_local + np.array([x, y])
            tile_centers_global = tile_centers_local + np.array([x, y])

            # A. Draw Grid Lines
            grid_9x9 = grid_points_global.reshape(9, 9, 2).astype(np.int32)
            for row in range(9):
                pts = np.ascontiguousarray(grid_9x9[row, :]).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], False, (255, 0, 0), 2)
            for col in range(9):
                pts = np.ascontiguousarray(grid_9x9[:, col]).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], False, (255, 0, 0), 2)

            # B. Process Pieces (Iterate list directly)
            for i, local_piece_mask in enumerate(piece_masks_list):
                
                # Moments calculation on ISOLATED mask
                M = cv2.moments(local_piece_mask)
                if M["m00"] == 0: continue
                
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Bias towards feet (approx 35% of bounding box height downwards)
                x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
                cY = int(cY + 0.35 * h_p)

                pc_local = np.array([cX, cY])
                pc_global = pc_local + np.array([x, y])

                # Unique Color
                hue = int((i * 137.508) % 180) 
                hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                piece_color = tuple(map(int, rgb_color))

                # Overlay
                global_single_piece_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
                global_single_piece_mask[y:y+h, x:x+w] = local_piece_mask

                colored_layer = np.zeros_like(vis_img)
                colored_layer[:] = piece_color

                mask_indices = global_single_piece_mask > 0
                if np.any(mask_indices):
                     vis_img[mask_indices] = cv2.addWeighted(
                        vis_img[mask_indices], 0.6, 
                        colored_layer[mask_indices], 0.4, 
                        0
                    )

                # Draw Vector
                dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
                closest_idx = np.argmin(dists)
                closest_center = tile_centers_global[closest_idx]
                
                pt1 = tuple(pc_global.astype(int))
                pt2 = tuple(closest_center.astype(int))
                
                cv2.line(vis_img, pt1, pt2, piece_color, 2)
                cv2.circle(vis_img, pt1, 6, (255, 255, 255), -1)
                cv2.circle(vis_img, pt1, 4, piece_color, -1)

            cv2.imwrite(os.path.join(output_dir, filename), vis_img)
            
            stats_total.append(total_ms)
            print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {pieces_ms:<8.1f} | {total_ms:<8.1f} | OK")

        except Exception as e:
            msg = f"ERROR: {str(e)}"
            print(f"{filename:<25} | {msg}")
            log_file.write(f"{filename}: {msg}\n")
            stats_skipped += 1

    log_file.close()
    
    print("-" * 95)
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
#     if not os.path.exists(output_dir): os.makedirs(output_dir)
    
#     log_path = os.path.join(output_dir, "filtered_frames.log")
#     log_file = open(log_path, "w")
#     print(f"Logging filtered frames to: {log_path}")

#     print(f"Initializing Engines from {assets_dir}...")
#     try:
#         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
#         solver = GridSolver()
#         # Quality Filter: Contact Ratio Check
#         quality_filter = QualityFilter(margin=1, min_score=0.5, max_border_contact_ratio=0.05)
#     except Exception as e:
#         print(f"CRITICAL ERROR: {e}")
#         sys.exit(1)

#     image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
#                          glob.glob(os.path.join(input_dir, '*.png')))
#     print(f"Found {len(image_files)} images.")

#     print("Warming up GPU...")
#     if os.path.exists('warmup.jpg'):
#         segmenter.predict('warmup.jpg', "chessboard")
#     else:
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         cv2.imwrite('warmup_dummy.jpg', dummy)
#         segmenter.predict('warmup_dummy.jpg', "chessboard")
#         os.remove('warmup_dummy.jpg')
#     print("Warmup done.\n")

#     stats_total = []
#     stats_skipped = 0
    
#     print(f"{'FILENAME':<25} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS'}")
#     print("-" * 95)

#     for img_path in image_files:
#         filename = os.path.basename(img_path)
#         t_start = time.perf_counter()
        
#         try:
#             # 1. Segmentation (Board)
#             mask, original_img, score = segmenter.predict(img_path, "chessboard")
#             t_seg_end = time.perf_counter()
#             seg_ms = (t_seg_end - t_start) * 1000.0

#             if seg_ms > TOTAL_TIME_LIMIT_MS:
#                 msg = "SKIP (Seg Slow)"
#                 print(f"{filename:<25} | {seg_ms:<8.1f} | {'-':<8} | {'-':<8} | {seg_ms:<8.1f} | {msg}")
#                 log_file.write(f"{filename}: {msg}\n")
#                 stats_skipped += 1
#                 continue

#             # 2. Filter
#             is_good, reason = quality_filter.check(mask, score)
#             if not is_good:
#                 msg = f"SKIP ({reason})"
#                 print(f"{filename:<25} | {seg_ms:<8.1f} | {'-':<8} | {'-':<8} | {seg_ms:<8.1f} | {msg}")
#                 log_file.write(f"{filename}: {msg}\n")
#                 stats_skipped += 1
#                 continue

#             # 3. Focus Step (Crop)
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
            
#             masked_full_img = cv2.bitwise_and(original_img, original_img, mask=mask)
#             crop_img = masked_full_img[y:y+h, x:x+w]
#             crop_mask = mask[y:y+h, x:x+w]

#             # 4. Grid Solving
#             remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
#             warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=remaining_ms)
            
#             t_grid_end = time.perf_counter()
#             grid_ms = (t_grid_end - t_seg_end) * 1000.0

#             if not success or 'timeout_at' in stats:
#                 status = f"SKIP (Timeout: {stats['timeout_at']})" if 'timeout_at' in stats else f"FAILED ({stats.get('error','')})"
#                 total_ms = (t_grid_end - t_start) * 1000.0
#                 print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {'-':<8} | {total_ms:<8.1f} | {status}")
#                 log_file.write(f"{filename}: {status}\n")
#                 stats_skipped += 1
#                 continue

#             # 5. Piece Detection (On Crop)
#             mask_pieces, _, _ = segmenter.predict(crop_img, "chess pieces")
#             t_pieces_end = time.perf_counter()
#             pieces_ms = (t_pieces_end - t_grid_end) * 1000.0
#             total_ms = (t_pieces_end - t_start) * 1000.0

#             # 6. Visualization & Mapping
#             vis_img = original_img.copy()
#             grid_points_global = grid_points_local + np.array([x, y])
#             tile_centers_global = tile_centers_local + np.array([x, y])

#             # A. Draw Grid Lines (Blue) - Draw these first so they are behind pieces
#             grid_9x9 = grid_points_global.reshape(9, 9, 2).astype(np.int32)
#             for row in range(9):
#                 pts = np.ascontiguousarray(grid_9x9[row, :]).reshape((-1, 1, 2))
#                 cv2.polylines(vis_img, [pts], False, (255, 0, 0), 2)
#             for col in range(9):
#                 pts = np.ascontiguousarray(grid_9x9[:, col]).reshape((-1, 1, 2))
#                 cv2.polylines(vis_img, [pts], False, (255, 0, 0), 2)

#             # B. Process Individual Pieces (Unique Colors)
#             # Get connected components (individual pieces) from the crop mask
#             num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_pieces)
            
#             # Iterate starting from 1 (0 is background)
#             for i in range(1, num_labels):
#                 area = stats[i, cv2.CC_STAT_AREA]
#                 if area < 50: continue # Skip noise

#                 # --- 1. Generate Unique Color ---
#                 # Use HSV to generate distinct colors based on the label ID
#                 # Hue varies based on ID, Saturation and Value are maxed for brightness
#                 hue = int((i * 137.508) % 180) # Use golden angle approximation for good spread
#                 hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
#                 # Convert to BGR for OpenCV
#                 rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
#                 piece_color = tuple(map(int, rgb_color))

#                 # --- 2. Create Individual Piece Mask & Overlay ---
#                 # Isolate mask for just this piece in local coordinates
#                 local_single_piece_mask = (labels == i).astype(np.uint8) * 255

#                 # Map to global coordinates
#                 global_single_piece_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
#                 global_single_piece_mask[y:y+h, x:x+w] = local_single_piece_mask

#                 # Create a solid color layer for this piece
#                 colored_layer = np.zeros_like(vis_img)
#                 colored_layer[:] = piece_color

#                 # Blend this specific piece onto the image
#                 mask_indices = global_single_piece_mask > 0
#                 if np.any(mask_indices):
#                      vis_img[mask_indices] = cv2.addWeighted(
#                         vis_img[mask_indices], 0.6, 
#                         colored_layer[mask_indices], 0.4, 
#                         0
#                     )

#                 # --- 3. Draw Vector to Tile Center ---
#                 pc_local = centroids[i]
#                 pc_global = pc_local + np.array([x, y])
                
#                 dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
#                 closest_idx = np.argmin(dists)
#                 closest_center = tile_centers_global[closest_idx]
                
#                 pt1 = tuple(pc_global.astype(int))
#                 pt2 = tuple(closest_center.astype(int))
                
#                 # Draw line and center dot using the unique piece color
#                 cv2.line(vis_img, pt1, pt2, piece_color, 2)
#                 # Add a white border to the dot for better contrast
#                 cv2.circle(vis_img, pt1, 6, (255, 255, 255), -1)
#                 cv2.circle(vis_img, pt1, 4, piece_color, -1)

#             cv2.imwrite(os.path.join(output_dir, filename), vis_img)
            
#             stats_total.append(total_ms)
#             print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {pieces_ms:<8.1f} | {total_ms:<8.1f} | OK")

#         except Exception as e:
#             msg = f"ERROR: {str(e)}"
#             print(f"{filename:<25} | {msg}")
#             log_file.write(f"{filename}: {msg}\n")
#             stats_skipped += 1

#     log_file.close()
    
#     print("-" * 95)
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

# # import os
# # import glob
# # import cv2
# # import numpy as np
# # import time
# # import sys
# # import argparse
# # from src.segmentation import ChessboardSegmenter
# # from src.grid_solver import GridSolver
# # from src.filter import QualityFilter

# # TOTAL_TIME_LIMIT_MS = 300.0 

# # def main(input_dir, output_dir, assets_dir):
# #     # Setup
# #     if not os.path.exists(output_dir): os.makedirs(output_dir)
    
# #     log_path = os.path.join(output_dir, "filtered_frames.log")
# #     log_file = open(log_path, "w")
# #     print(f"Logging filtered frames to: {log_path}")

# #     print(f"Initializing Engines from {assets_dir}...")
# #     try:
# #         # Load BOTH prompts
# #         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
# #         solver = GridSolver()
        
# #         # FILTER FIX: Relaxed threshold to 0.15 (15%)
# #         # This allows corners/sides to touch the edge slightly without skipping
# #         quality_filter = QualityFilter(margin=1, min_score=0.5, max_border_contact_ratio=0.05)
        
# #     except Exception as e:
# #         print(f"CRITICAL ERROR: {e}")
# #         sys.exit(1)

# #     image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
# #                          glob.glob(os.path.join(input_dir, '*.png')))
# #     print(f"Found {len(image_files)} images.")

# #     print("Warming up GPU...")
# #     if os.path.exists('warmup.jpg'):
# #         segmenter.predict('warmup.jpg', "chessboard")
# #     else:
# #         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
# #         cv2.imwrite('warmup_dummy.jpg', dummy)
# #         segmenter.predict('warmup_dummy.jpg', "chessboard")
# #         os.remove('warmup_dummy.jpg')
# #     print("Warmup done.\n")

# #     stats_total = []
# #     stats_skipped = 0
    
# #     print(f"{'FILENAME':<25} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS'}")
# #     print("-" * 95)

# #     for img_path in image_files:
# #         filename = os.path.basename(img_path)
# #         t_start = time.perf_counter()
        
# #         try:
# #             # 1. Segmentation (Board)
# #             mask, original_img, score = segmenter.predict(img_path, "chessboard")
            
# #             t_seg_end = time.perf_counter()
# #             seg_ms = (t_seg_end - t_start) * 1000.0

# #             if seg_ms > TOTAL_TIME_LIMIT_MS:
# #                 msg = "SKIP (Seg Slow)"
# #                 print(f"{filename:<25} | {seg_ms:<8.1f} | {'-':<8} | {'-':<8} | {seg_ms:<8.1f} | {msg}")
# #                 log_file.write(f"{filename}: {msg}\n")
# #                 stats_skipped += 1
# #                 continue

# #             # 2. Quality Filter
# #             is_good, reason = quality_filter.check(mask, score)
# #             if not is_good:
# #                 msg = f"SKIP ({reason})"
# #                 print(f"{filename:<25} | {seg_ms:<8.1f} | {'-':<8} | {'-':<8} | {seg_ms:<8.1f} | {msg}")
# #                 log_file.write(f"{filename}: {msg}\n")
# #                 stats_skipped += 1
# #                 continue

# #             # --- 3. FOCUS STEP ---
# #             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #             largest_contour = max(contours, key=cv2.contourArea)
# #             x, y, w, h = cv2.boundingRect(largest_contour)
            
# #             # Mask & Crop
# #             masked_full_img = cv2.bitwise_and(original_img, original_img, mask=mask)
# #             crop_img = masked_full_img[y:y+h, x:x+w]
# #             crop_mask = mask[y:y+h, x:x+w]

# #             # 4. Grid Solving (On Crop)
# #             remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
# #             warped_grid, corners_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=remaining_ms)
            
# #             t_grid_end = time.perf_counter()
# #             grid_ms = (t_grid_end - t_seg_end) * 1000.0

# #             if not success or 'timeout_at' in stats:
# #                 status = f"SKIP (Timeout: {stats['timeout_at']})" if 'timeout_at' in stats else f"FAILED ({stats.get('error','')})"
# #                 total_ms = (t_grid_end - t_start) * 1000.0
# #                 print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {'-':<8} | {total_ms:<8.1f} | {status}")
# #                 log_file.write(f"{filename}: {status}\n")
# #                 stats_skipped += 1
# #                 continue

# #             # 5. Piece Detection
# #             mask_pieces, _, _ = segmenter.predict(crop_img, "chess pieces")
            
# #             t_pieces_end = time.perf_counter()
# #             pieces_ms = (t_pieces_end - t_grid_end) * 1000.0
# #             total_ms = (t_pieces_end - t_start) * 1000.0

# #             # 6. Visualization & Save
# #             corners_global = corners_local + np.array([x, y])
            
# #             vis_img = original_img.copy()
            
# #             # Draw Grid (Green)
# #             cv2.polylines(vis_img, [corners_global.astype(np.int32)], True, (0, 255, 0), 3)
            
# #             # Draw Pieces (Red Overlay)
# #             global_piece_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
# #             global_piece_mask[y:y+h, x:x+w] = mask_pieces
            
# #             # --- FIX FOR OPENCV CRASH ---
# #             # Instead of blending vectors, we create a full red layer and blend slices
# #             if np.count_nonzero(global_piece_mask) > 0:
# #                 # 1. Create Red Layer
# #                 red_layer = np.zeros_like(vis_img)
# #                 red_layer[:] = (0, 0, 255) # BGR for Red
                
# #                 # 2. Get indices
# #                 mask_indices = global_piece_mask > 0
                
# #                 # 3. Blend exactly matching shapes (N,3) with (N,3)
# #                 vis_img[mask_indices] = cv2.addWeighted(
# #                     vis_img[mask_indices], 0.5, 
# #                     red_layer[mask_indices], 0.5, 
# #                     0
# #                 )

# #             cv2.imwrite(os.path.join(output_dir, filename), vis_img)
            
# #             stats_total.append(total_ms)
# #             print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {pieces_ms:<8.1f} | {total_ms:<8.1f} | OK")

# #         except Exception as e:
# #             # Print full error details to debug
# #             import traceback
# #             msg = f"ERROR: {str(e)}"
# #             # traceback.print_exc() # Uncomment if you need stack trace
# #             print(f"{filename:<25} | {msg}")
# #             log_file.write(f"{filename}: {msg}\n")
# #             stats_skipped += 1

# #     log_file.close()
    
# #     print("-" * 95)
# #     print(f"Processed (OK): {len(stats_total)}")
# #     print(f"Skipped / Failed: {stats_skipped}")
# #     if stats_total:
# #         print(f"Average Latency (OK): {sum(stats_total)/len(stats_total):.2f} ms")

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--input", default="../vid1", help="Input directory")
# #     parser.add_argument("--output", default="./output", help="Output directory")
# #     parser.add_argument("--assets", default="./assets", help="Assets directory")
# #     args = parser.parse_args()
    
# #     main(args.input, args.output, args.assets)