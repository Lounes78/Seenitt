"""
Chessboard Detection Pipeline with Performance Optimizations:

1. Vision Encoder Caching: Run the expensive Vision Encoder once per frame,
   then reuse features for both board and piece segmentation (decoder only).
   
2. Parallel CPU/GPU Execution: Grid Solver (CPU) and Piece Decoder (GPU) 
   run simultaneously using ThreadPool, hiding latency of the faster operation.

3. Temporal Grid Tracking: Use optical flow to track grid across frames,
   falling back to full solver when tracking fails.
"""

import os
import glob
import cv2
import numpy as np
import time
import sys
import argparse
from multiprocessing.pool import ThreadPool
import pycuda.autoinit

from src.segmentation import ChessboardSegmenter
from src.grid_solver import GridSolver
from src.filter import QualityFilter
from src.grid_tracker import GridTracker
from src.piece_processor import process_pieces
from src.visualizer import draw_grid_lines, visualize_pieces

TOTAL_TIME_LIMIT_MS = 500.0

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
        vision_feats, orig = segmenter.encode_image('warmup.jpg')
        segmenter.decode_from_features(vision_feats, orig, "chessboard")
        segmenter.decode_from_features(vision_feats, orig, "chess pieces")
    else:
        dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
        cv2.imwrite('warmup_dummy.jpg', dummy)
        vision_feats, orig = segmenter.encode_image('warmup_dummy.jpg')
        segmenter.decode_from_features(vision_feats, orig, "chessboard")
        segmenter.decode_from_features(vision_feats, orig, "chess pieces")
        os.remove('warmup_dummy.jpg')
    print("Warmup done.\n")

    stats_total = []
    stats_skipped = 0
    
    # Initialize temporal grid tracker
    tracker = GridTracker()
    
    print(f"{'FILENAME':<25} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS'}")
    print("-" * 95)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        t_start = time.perf_counter()
        
        try:
            # 1. Encode Image ONCE (Vision Encoder)
            vision_features, original_img = segmenter.encode_image(img_path)
            t_encode_end = time.perf_counter()
            encode_ms = (t_encode_end - t_start) * 1000.0

            if encode_ms > TOTAL_TIME_LIMIT_MS:
                msg = "SKIP (Encode Slow)"
                print(f"{filename:<25} | {encode_ms:<8.1f} | {'-':<8} | {'-':<8} | {encode_ms:<8.1f} | {msg}")
                log_file.write(f"{filename}: {msg}\n")
                stats_skipped += 1
                continue

            # 2. Decode Board (using cached vision features)
            mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
            t_seg_end = time.perf_counter()
            seg_ms = (t_seg_end - t_start) * 1000.0

            # 3. Filter
            is_good, reason = quality_filter.check(mask, score)
            if not is_good:
                msg = f"SKIP ({reason})"
                print(f"{filename:<25} | {seg_ms:<8.1f} | {'-':<8} | {'-':<8} | {seg_ms:<8.1f} | {msg}")
                log_file.write(f"{filename}: {msg}\n")
                stats_skipped += 1
                continue

            # 4. Focus Step (Crop)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Use original image for crop (don't mask background to black)
            # This prevents cutting off pieces that stick out of the board mask
            crop_img = original_img[y:y+h, x:x+w].copy()
            crop_mask = mask[y:y+h, x:x+w]
            curr_crop_bbox = (x, y, w, h)

            # --- TEMPORAL GRID TRACKING ---
            curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            # Try tracking first
            tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(
                curr_gray, curr_crop_bbox
            )
            
            # --- PARALLEL EXECUTION: Grid Solver (CPU) || Piece Decoder (GPU) ---
            
            # Define wrapper for piece segmentation to run in background thread
            def run_piece_decoder():
                # 1. Get the global CUDA context
                ctx = pycuda.autoinit.context
                # 2. Push it to this thread so it can see the GPU
                ctx.push()
                try:
                    return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
                except Exception as e:
                    print(f"GPU Thread Error: {e}")
                    raise e
                finally:
                    # 3. Pop it to clean up
                    ctx.pop()
            
            # Launch Piece Decoder on GPU in background thread
            pool = ThreadPool(processes=1)
            async_pieces = pool.apply_async(run_piece_decoder)
            
            # Run Grid Solver on CPU in main thread (parallel with GPU decoder)
            # SKIP if we already tracked the grid
            t_parallel_start = time.perf_counter()
            success = tracking_success
            
            if not tracking_success:
                # SLOW: Full solver (~150ms)
                remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
                warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(
                    crop_img, crop_mask, time_limit_ms=remaining_ms
                )
                
                if success:
                    # Enable tracking for next frame
                    tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
            
            t_grid_end = time.perf_counter()
            grid_ms = (t_grid_end - t_parallel_start) * 1000.0

            if not success or 'timeout_at' in stats:
                async_pieces.wait()  # Ensure GPU task completes before cleanup
                pool.close()
                pool.join()
                status = f"SKIP (Timeout: {stats['timeout_at']})" if 'timeout_at' in stats else f"FAILED ({stats.get('error','')})"
                total_ms = (t_grid_end - t_start) * 1000.0
                print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {'-':<8} | {total_ms:<8.1f} | {status}")
                log_file.write(f"{filename}: {status}\n")
                stats_skipped += 1
                
                # Reset tracking on failure
                tracker.reset()
                continue
            
            # Wait for Piece Decoder to finish and retrieve results
            piece_masks_list, _, _ = async_pieces.get()
            pool.close()
            pool.join()
            
            t_pieces_end = time.perf_counter()
            pieces_ms = max((t_pieces_end - t_parallel_start) * 1000.0 - grid_ms, 0.0)
            total_ms = (t_pieces_end - t_start) * 1000.0
            
            # --- END PARALLEL EXECUTION ---

            # 7. Process Pieces and Visualization
            vis_img = original_img.copy()
            grid_points_global = grid_points_local + np.array([x, y])
            tile_centers_global = tile_centers_local + np.array([x, y])

            # A. Draw Grid Lines
            draw_grid_lines(vis_img, grid_points_global)
            
            # B. Process and Deduplicate Pieces
            tile_to_piece = process_pieces(
                piece_masks_list,
                curr_crop_bbox,
                grid_points_global,
                tile_centers_global
            )
            
            # C. Visualize Pieces
            visualize_pieces(vis_img, tile_to_piece, tile_centers_global)

            cv2.imwrite(os.path.join(output_dir, filename), vis_img)
            
            stats_total.append(total_ms)
            status_suffix = " [TRACKED]" if tracking_success else ""
            print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {pieces_ms:<8.1f} | {total_ms:<8.1f} | OK{status_suffix}")

        except Exception as e:
            msg = f"ERROR: {str(e)}"
            print(f"{filename:<25} | {msg}")
            log_file.write(f"{filename}: {msg}\n")
            stats_skipped += 1
            
            # Reset tracking on error
            tracker.reset()

    log_file.close()
    
    print("-" * 95)
    print(f"Processed (OK): {len(stats_total)}")
    print(f"Skipped / Failed: {stats_skipped}")
    if stats_total:
        print(f"Average Latency (OK): {sum(stats_total)/len(stats_total):.2f} ms")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./captured_frames", help="Input directory")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--assets", default="./assets", help="Assets directory")
    args = parser.parse_args()
    
    main(args.input, args.output, args.assets)
