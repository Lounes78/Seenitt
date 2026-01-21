# src/server/worker.py

import os
import time
import traceback
import cv2
import numpy as np
import pycuda.autoinit
from multiprocessing.pool import ThreadPool

# Import Engines
from src.segmentation import ChessboardSegmenter
from src.grid_solver import GridSolver
from src.filter import QualityFilter
from src.grid_tracker import GridTracker
from src.piece_processor import process_pieces
from src.visualizer import draw_grid_lines, visualize_pieces

# Import Server Utils
from src.server.config import TOTAL_TIME_LIMIT_MS, SAVE_DEBUG_FRAMES, OUTPUT_DIR
from src.server.utils import print_ascii_board

def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
    """
    Main AI processing loop.
    Reads frames from a Queue to ensure data integrity.
    """
    print("[AI] Worker Started.", flush=True)
    
    # 1. Initialize Engines
    try:
        if SAVE_DEBUG_FRAMES:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print(f"[AI] Initializing Engines from {assets_dir}...", flush=True)
        
        segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
        solver = GridSolver()
        quality_filter = QualityFilter(margin=1, min_score=0.5, max_border_contact_ratio=0.05)
        tracker = GridTracker()
        
        print("[AI] Engines Loaded.", flush=True)
        
        # GPU Warmup
        print("[AI] Warming up GPU...", flush=True)
        dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
        vision_feats, orig = segmenter.encode_image(dummy)
        segmenter.decode_from_features(vision_feats, orig, "chessboard")
        segmenter.decode_from_features(vision_feats, orig, "chess pieces")
        print("[AI] Warmup done.\n", flush=True)
        
    except Exception as e:
        print(f"[AI] Critical Init Failed: {e}", flush=True)
        traceback.print_exc()
        return

    stats_total = []
    stats_skipped = 0

    print(f"{'FRAME':<10} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS'}")
    print("-" * 85)

    while True:
        try:
            try:
                packet = frame_queue.get(timeout=1.0)
            except:
                continue 
            
            if packet is None:
                print("[AI] Received kill signal, exiting.", flush=True)
                break
                
            frame_id, frame = packet
            
            # Ensure C-Contiguous memory for C++ bindings
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
                
            t_start = time.perf_counter()

            # --- PIPELINE START ---
            vision_features, original_img = segmenter.encode_image(frame)
            t_encode_end = time.perf_counter()
            encode_ms = (t_encode_end - t_start) * 1000.0

            if encode_ms > TOTAL_TIME_LIMIT_MS:
                print(f"Frame {frame_id:<10} | {encode_ms:<8.1f} | - | - | {encode_ms:<8.1f} | SKIP (Encode Slow)")
                stats_skipped += 1
                continue

            mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
            t_seg_end = time.perf_counter()
            seg_ms = (t_seg_end - t_start) * 1000.0

            is_good, reason = quality_filter.check(mask, score)
            if not is_good:
                print(f"Frame {frame_id:<10} | {seg_ms:<8.1f} | - | - | {seg_ms:<8.1f} | SKIP ({reason})")
                stats_skipped += 1
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                stats_skipped += 1
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            if w < 50 or h < 50:
                continue

            crop_img = original_img[y:y+h, x:x+w].copy()
            crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
            curr_crop_bbox = (x, y, w, h)
            curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            # Tracker
            tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
            
            # Parallel GPU/CPU
            def run_piece_decoder():
                ctx = pycuda.autoinit.context
                ctx.push()
                try:
                    return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
                finally:
                    ctx.pop()
            
            pool = ThreadPool(processes=1)
            async_pieces = pool.apply_async(run_piece_decoder)
            
            t_parallel_start = time.perf_counter()
            success = tracking_success
            
            if not tracking_success:
                t_spent = (time.perf_counter() - t_start) * 1000
                remaining_ms = TOTAL_TIME_LIMIT_MS - t_spent
                
                warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(
                    crop_img, crop_mask, time_limit_ms=remaining_ms
                )
                
                if success:
                    tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
            
            t_grid_end = time.perf_counter()
            grid_ms = (t_grid_end - t_parallel_start) * 1000.0

            if not success or 'timeout_at' in stats:
                async_pieces.wait()
                pool.close()
                pool.join()
                stats_skipped += 1
                tracker.reset()
                status = "TIMEOUT" if 'timeout_at' in stats else "FAILED"
                print(f"Frame {frame_id:<10} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | - | {(t_grid_end-t_start)*1000:<8.1f} | {status}")
                continue
            
            piece_masks_list, _, _ = async_pieces.get()
            pool.close()
            pool.join()
            
            t_pieces_end = time.perf_counter()
            pieces_ms = max((t_pieces_end - t_parallel_start) * 1000.0 - grid_ms, 0.0)
            total_ms = (t_pieces_end - t_start) * 1000.0

            # Visualization
            vis_img = original_img.copy()
            grid_points_global = grid_points_local + np.array([x, y])
            tile_centers_global = tile_centers_local + np.array([x, y])

            draw_grid_lines(vis_img, grid_points_global)
            tile_to_piece = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)
            visualize_pieces(vis_img, tile_to_piece, tile_centers_global)

            if SAVE_DEBUG_FRAMES:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)
            
            stats_total.append(total_ms)
            status_suffix = " [TRACKED]" if tracking_success else ""
            print(f"Frame {frame_id:<10} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {pieces_ms:<8.1f} | {total_ms:<8.1f} | OK{status_suffix}")
            
            if tile_to_piece:
                print_ascii_board(tile_to_piece, frame_id)
            
            if frame_id % 100 == 0 and len(stats_total) > 0:
                avg = sum(stats_total) / len(stats_total)
                result_queue.put_nowait(f"Status: {len(stats_total)} OK, {stats_skipped} skipped, {avg:.1f}ms avg")

        except Exception as e:
            print(f"[AI] Error: {e}", flush=True)
            traceback.print_exc()
            tracker.reset()
            time.sleep(0.1)
