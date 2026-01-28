import os
import time
import traceback
import cv2
import numpy as np
import pycuda.autoinit
from multiprocessing.pool import ThreadPool

# --- CHESS LIBRARY IMPORTS ---
import chess
import chess.engine

from src.segmentation import ChessboardSegmenter
from src.grid_solver import GridSolver
from src.filter import QualityFilter
from src.grid_tracker import GridTracker
from src.piece_processor import process_pieces
from src.visualizer import draw_grid_lines, visualize_pieces
from src.server.config import TOTAL_TIME_LIMIT_MS, SAVE_DEBUG_FRAMES, OUTPUT_DIR
from src.server.utils import print_ascii_board
from src.stabilizer import PositionStabilizer 

# --- HELPER: ORIENTATION CORRECTION ---
def correct_orientation(grid_points, tile_centers):
    g = grid_points.reshape(9, 9, 2)
    c = tile_centers.reshape(8, 8, 2)
    best_dist = float('inf')
    best_k = 0
    for k in range(4):
        rotated_g = np.rot90(g, k)
        p0 = rotated_g[0, 0]
        dist = (p0[0] ** 2) + (p0[1] ** 2)
        if dist < best_dist:
            best_dist = dist
            best_k = k
    g = np.rot90(g, best_k)
    c = np.rot90(c, best_k)
    p0, p1 = g[0, 0], g[0, 1]
    dx, dy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
    if dy > dx:
        g = np.transpose(g, (1, 0, 2))
        c = np.transpose(c, (1, 0, 2))
    return np.ascontiguousarray(g.reshape(-1, 2)), np.ascontiguousarray(c.reshape(-1, 2))

# --- HELPER: FEN GENERATOR ---
def generate_fen(ascii_map):
    fen = ""
    for row in range(8):
        empty = 0
        for col in range(8):
            idx = row * 8 + col
            char = ascii_map.get(idx, '.')
            if char == '.':
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += char
        if empty > 0: fen += str(empty)
        if row < 7: fen += "/"
    return fen

def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
    print("[AI] Worker Started.", flush=True)
    
    # --- STOCKFISH CONFIGURATION ---
    STOCKFISH_PATH = "/home/matcha/seenit/Seenitt/core/src/server/stockfish/stockfish-ubuntu-x86-64-avx2"
    engine = None
    
    # --- STATE TRACKING ---
    last_fen_analyzed = None 

    try:
        if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
        solver = GridSolver()
        quality_filter = QualityFilter(margin=5, min_score=0.5)
        tracker = GridTracker()
        stabilizer = PositionStabilizer()

        try:
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            print("[AI] Stockfish Engine Loaded.", flush=True)
        except Exception as e:
            print(f"[AI] Engine Warning: Could not load Stockfish ({e}). Advice will be disabled.")

        # Warmup
        dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
        f, o = segmenter.encode_image(dummy)
        segmenter.decode_from_features(f, o, "chessboard")
        segmenter.decode_from_features(f, o, "chess pieces")
        print("[AI] Ready.", flush=True)

    except Exception as e:
        print(f"[AI] Init Failed: {e}", flush=True)
        traceback.print_exc()
        return

    stats_total = []
    stats_skipped = 0

    print(f"{'FRAME':<6} | {'ENC':<6} | {'SEG':<6} | {'GRID':<6} | {'PIECE':<6} | {'TOTAL':<6}")

    try:
        while True:
            try:
                try: packet = frame_queue.get(timeout=1.0)
                except: continue 
                if packet is None: break
                frame_id, frame = packet
                if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
                
                t_start = time.perf_counter()

                # 1. Encode
                vision_features, original_img = segmenter.encode_image(frame)
                t_encode = time.perf_counter()

                if (t_encode - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
                    stats_skipped += 1; continue

                # 2. Segment Board
                mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
                is_good, _ = quality_filter.check(mask, score)
                if not is_good: stats_skipped += 1; continue

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w < 50 or h < 50: continue

                crop_img = original_img[y:y+h, x:x+w].copy()
                crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
                curr_crop_bbox = (x, y, w, h)
                curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                
                t_seg = time.perf_counter()
                
                # 3. Track / Solve Grid
                tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
                
                def run_pieces():
                    ctx = pycuda.autoinit.context
                    ctx.push()
                    try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
                    finally: ctx.pop()
                
                pool = ThreadPool(processes=1)
                async_pieces = pool.apply_async(run_pieces)
                
                success = tracking_success
                if not tracking_success:
                    rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
                    warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
                    if success: tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
                
                t_grid = time.perf_counter()

                if not success:
                    async_pieces.wait(); pool.close(); pool.join()
                    stats_skipped += 1; tracker.reset()
                    continue

                grid_points_local, tile_centers_local = correct_orientation(grid_points_local, tile_centers_local)
                piece_masks_list, _, _ = async_pieces.get()
                pool.close(); pool.join()

                # 4. Process Pieces & Stabilize
                grid_points_global = grid_points_local + np.array([x, y])
                tile_centers_global = tile_centers_local + np.array([x, y])
                
                detected_tiles_map = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)
                raw_indices = list(detected_tiles_map.keys())
                stable_ascii_map = stabilizer.update(raw_indices)
                
                t_end = time.perf_counter()
                ms_enc    = (t_encode - t_start) * 1000
                ms_seg    = (t_seg - t_encode) * 1000
                ms_grid   = (t_grid - t_seg) * 1000
                ms_piece  = (t_end - t_grid) * 1000
                ms_total  = (t_end - t_start) * 1000
                stats_total.append(ms_total)

                # --- VISUALIZATION ---
                print(f"\n[{frame_id:<5}] {ms_enc:.0f}ms enc | {ms_seg:.0f}ms seg | {ms_grid:.0f}ms grid | {ms_piece:.0f}ms piece | = {ms_total:.0f}ms total")
                print(f"{'RAW DETECTIONS (Left)':<25} | {'STABILIZED (Right)':<25}")
                print("-" * 75)
                for row in range(8):
                    line_raw = ""
                    for col in range(8):
                        idx = row * 8 + col
                        char = 'X' if idx in raw_indices else '.'
                        line_raw += f"{char} "
                    line_stable = ""
                    for col in range(8):
                        idx = row * 8 + col
                        char = stable_ascii_map.get(idx, '.')
                        line_stable += f"{char} "
                    print(f"Row {8-row}       | {line_raw:<25} | {line_stable:<25}")
                print("-" * 75)

                # --- GET ADVICE (Only on YOUR turn) ---
                current_fen = generate_fen(stable_ascii_map)
                best_move_uci = None
                
                # Check 1: Engine loaded?
                # Check 2: Board state changed?
                # Check 3: Is it WHITE's turn? (Assuming User = White)
                is_user_turn = (stabilizer.board.turn == chess.WHITE)

                if engine and current_fen != last_fen_analyzed and is_user_turn:
                    try:
                        # We use the stabilizer's internal board directly because it has the correct turn info
                        # (current_fen from generate_fen is purely visual and lacks turn info 'w' or 'b')
                        
                        # Use stabilizer.board for analysis to ensure correct turn
                        result = engine.play(stabilizer.board, chess.engine.Limit(time=0.1))
                        
                        if result.move:
                            best_move_uci = result.move.uci()
                            last_fen_analyzed = current_fen
                            print(f"[AI] 💡 YOUR TURN ({stabilizer.board.fullmove_number}): Advice -> {best_move_uci}")
                    except Exception as err:
                        print(f"[AI] Engine Error: {err}")
                
                elif not is_user_turn:
                     # Update the FEN tracker so we don't re-analyze immediately when turn passes back
                     # But don't give advice
                     last_fen_analyzed = current_fen


                
                # Create base payload
                payload = {
                    "type": "board_state", 
                    "fen": current_fen, 
                    "frame_id": frame_id
                }
                
                # Only insert advice if it is NOT None
                if best_move_uci:
                    payload["advice"] = best_move_uci
                
                result_queue.put(payload)

                
                # result_queue.put({
                #     "type": "board_state", 
                #     "fen": current_fen, 
                #     "frame_id": frame_id,
                #     "advice": best_move_uci
                # })
                
                if SAVE_DEBUG_FRAMES:
                    vis_img = original_img.copy()
                    draw_grid_lines(vis_img, grid_points_global)
                    visualize_pieces(vis_img, detected_tiles_map, tile_centers_global)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)

            except Exception as e:
                print(f"[AI] Error: {e}", flush=True)
                traceback.print_exc()
                tracker.reset()
                time.sleep(0.1)

    finally:
        if engine:
            engine.quit()
            print("[AI] Engine stopped.")










# import os
# import time
# import traceback
# import cv2
# import numpy as np
# import pycuda.autoinit
# from multiprocessing.pool import ThreadPool

# # --- NEW: Chess Library Imports ---
# import chess
# import chess.engine

# # --- EXISTING IMPORTS ---
# from src.segmentation import ChessboardSegmenter
# from src.grid_solver import GridSolver
# from src.filter import QualityFilter
# from src.grid_tracker import GridTracker
# from src.piece_processor import process_pieces
# from src.visualizer import draw_grid_lines, visualize_pieces
# from src.server.config import TOTAL_TIME_LIMIT_MS, SAVE_DEBUG_FRAMES, OUTPUT_DIR
# from src.server.utils import print_ascii_board
# from src.stabilizer import PositionStabilizer 

# # --- HELPER: ORIENTATION CORRECTION ---
# def correct_orientation(grid_points, tile_centers):
#     g = grid_points.reshape(9, 9, 2)
#     c = tile_centers.reshape(8, 8, 2)
#     best_dist = float('inf')
#     best_k = 0
#     for k in range(4):
#         rotated_g = np.rot90(g, k)
#         p0 = rotated_g[0, 0]
#         dist = (p0[0] ** 2) + (p0[1] ** 2)
#         if dist < best_dist:
#             best_dist = dist
#             best_k = k
#     g = np.rot90(g, best_k)
#     c = np.rot90(c, best_k)
#     p0, p1 = g[0, 0], g[0, 1]
#     dx, dy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
#     if dy > dx:
#         g = np.transpose(g, (1, 0, 2))
#         c = np.transpose(c, (1, 0, 2))
#     return np.ascontiguousarray(g.reshape(-1, 2)), np.ascontiguousarray(c.reshape(-1, 2))

# # --- HELPER: FEN GENERATOR ---
# def generate_fen(ascii_map):
#     fen = ""
#     for row in range(8):
#         empty = 0
#         for col in range(8):
#             idx = row * 8 + col
#             char = ascii_map.get(idx, '.')
#             if char == '.':
#                 empty += 1
#             else:
#                 if empty > 0:
#                     fen += str(empty)
#                     empty = 0
#                 fen += char
#         if empty > 0: fen += str(empty)
#         if row < 7: fen += "/"
#     return fen





# def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
#     print("[AI] Worker Started.", flush=True)
    
#     # --- STOCKFISH CONFIGURATION ---
#     STOCKFISH_PATH = "/home/matcha/seenit/Seenitt/core/src/server/stockfish/stockfish-ubuntu-x86-64-avx2"
#     engine = None
    
#     # --- STATE TRACKING ---
#     last_fen_analyzed = None 

#     try:
#         if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         # Initialize Computer Vision components
#         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
#         solver = GridSolver()
#         quality_filter = QualityFilter(margin=5, min_score=0.5)
#         tracker = GridTracker()
#         stabilizer = PositionStabilizer()

#         # --- INITIALIZE ENGINE ---
#         try:
#             engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
#             print("[AI] Stockfish Engine Loaded.", flush=True)
#         except Exception as e:
#             print(f"[AI] Engine Warning: Could not load Stockfish ({e}). Advice will be disabled.")

#         # Warmup
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         f, o = segmenter.encode_image(dummy)
#         segmenter.decode_from_features(f, o, "chessboard")
#         segmenter.decode_from_features(f, o, "chess pieces")
#         print("[AI] Ready.", flush=True)

#     except Exception as e:
#         print(f"[AI] Init Failed: {e}", flush=True)
#         traceback.print_exc()
#         return

#     stats_total = []
#     stats_skipped = 0

#     print(f"{'FRAME':<6} | {'ENC':<6} | {'SEG':<6} | {'GRID':<6} | {'PIECE':<6} | {'TOTAL':<6}")

#     try:
#         while True:
#             try:
#                 try: packet = frame_queue.get(timeout=1.0)
#                 except: continue 
#                 if packet is None: break
#                 frame_id, frame = packet
#                 if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
                
#                 t_start = time.perf_counter()

#                 # 1. Encode
#                 vision_features, original_img = segmenter.encode_image(frame)
#                 t_encode = time.perf_counter()

#                 if (t_encode - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
#                     stats_skipped += 1; continue

#                 # 2. Segment Board
#                 mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
#                 is_good, _ = quality_filter.check(mask, score)
#                 if not is_good: stats_skipped += 1; continue

#                 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 if not contours: continue
#                 largest_contour = max(contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
#                 if w < 50 or h < 50: continue

#                 crop_img = original_img[y:y+h, x:x+w].copy()
#                 crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
#                 curr_crop_bbox = (x, y, w, h)
#                 curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                
#                 t_seg = time.perf_counter()
                
#                 # 3. Track / Solve Grid
#                 tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
                
#                 def run_pieces():
#                     ctx = pycuda.autoinit.context
#                     ctx.push()
#                     try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
#                     finally: ctx.pop()
                
#                 pool = ThreadPool(processes=1)
#                 async_pieces = pool.apply_async(run_pieces)
                
#                 success = tracking_success
#                 if not tracking_success:
#                     rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
#                     warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
#                     if success: tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
                
#                 t_grid = time.perf_counter()

#                 if not success:
#                     async_pieces.wait(); pool.close(); pool.join()
#                     stats_skipped += 1; tracker.reset()
#                     # print(f"Frame {frame_id} FAILED: {stats}")
#                     continue

#                 grid_points_local, tile_centers_local = correct_orientation(grid_points_local, tile_centers_local)
#                 piece_masks_list, _, _ = async_pieces.get()
#                 pool.close(); pool.join()

#                 # 4. Process Pieces & Stabilize
#                 grid_points_global = grid_points_local + np.array([x, y])
#                 tile_centers_global = tile_centers_local + np.array([x, y])
                
#                 detected_tiles_map = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)
#                 raw_indices = list(detected_tiles_map.keys())
#                 stable_ascii_map = stabilizer.update(raw_indices)
                
#                 t_end = time.perf_counter()

#                 # --- CALCULATE TIMINGS ---
#                 ms_enc    = (t_encode - t_start) * 1000
#                 ms_seg    = (t_seg - t_encode) * 1000
#                 ms_grid   = (t_grid - t_seg) * 1000
#                 ms_piece  = (t_end - t_grid) * 1000
#                 ms_total  = (t_end - t_start) * 1000
#                 stats_total.append(ms_total)

#                 # --- TERMINAL VISUALIZATION (RESTORED) ---
#                 print(f"\n[{frame_id:<5}] {ms_enc:.0f}ms enc | {ms_seg:.0f}ms seg | {ms_grid:.0f}ms grid | {ms_piece:.0f}ms piece | = {ms_total:.0f}ms total")
#                 print(f"{'RAW DETECTIONS (Left)':<25} | {'STABILIZED (Right)':<25}")
#                 print("-" * 75)
#                 for row in range(8):
#                     # Build Raw Line
#                     line_raw = ""
#                     for col in range(8):
#                         idx = row * 8 + col
#                         char = 'X' if idx in raw_indices else '.'
#                         line_raw += f"{char} "
                    
#                     # Build Stable Line
#                     line_stable = ""
#                     for col in range(8):
#                         idx = row * 8 + col
#                         char = stable_ascii_map.get(idx, '.')
#                         line_stable += f"{char} "
                    
#                     print(f"Row {8-row}       | {line_raw:<25} | {line_stable:<25}")
#                 print("-" * 75)

#                 # --- GET ADVICE FROM ENGINE (STATE CONTROLLED) ---
#                 current_fen = generate_fen(stable_ascii_map)
#                 best_move_uci = None
                
#                 # Check if FEN changed or if it's the very first valid FEN
#                 if engine and current_fen != last_fen_analyzed:
#                     try:
#                         # Append Turn info if missing
#                         fen_to_check = current_fen
#                         if " " not in fen_to_check:
#                              fen_to_check += " w - - 0 1"

#                         board = chess.Board(fen_to_check)
#                         if board.is_valid():
#                             # Analyze
#                             result = engine.play(board, chess.engine.Limit(time=0.1))
#                             if result.move:
#                                 best_move_uci = result.move.uci()
#                                 last_fen_analyzed = current_fen # Update state
#                                 print(f"[AI] 💡 New Advice Generated: {best_move_uci}")
#                     except Exception as err:
#                         pass
                
#                 # --- SEND RESULT ---
#                 result_queue.put({
#                     "type": "board_state", 
#                     "fen": current_fen, 
#                     "frame_id": frame_id,
#                     "advice": best_move_uci 
#                 })
                
#                 if SAVE_DEBUG_FRAMES:
#                     vis_img = original_img.copy()
#                     draw_grid_lines(vis_img, grid_points_global)
#                     visualize_pieces(vis_img, detected_tiles_map, tile_centers_global)
#                     cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)

#             except Exception as e:
#                 print(f"[AI] Error: {e}", flush=True)
#                 traceback.print_exc()
#                 tracker.reset()
#                 time.sleep(0.1)

#     finally:
#         if engine:
#             engine.quit()
#             print("[AI] Engine stopped.")















            
# def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
#     print("[AI] Worker Started.", flush=True)
    
#     # --- STOCKFISH CONFIGURATION ---
#     STOCKFISH_PATH = "/home/matcha/seenit/Seenitt/core/src/server/stockfish/stockfish-ubuntu-x86-64-avx2"
#     engine = None
    
#     # --- NEW: TRACKING STATE ---
#     last_fen_analyzed = None 

#     try:
#         if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         # Initialize Computer Vision components
#         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
#         solver = GridSolver()
#         quality_filter = QualityFilter(margin=5, min_score=0.5)
#         tracker = GridTracker()
#         stabilizer = PositionStabilizer()

#         # --- INITIALIZE ENGINE ---
#         try:
#             engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
#             print("[AI] Stockfish Engine Loaded.", flush=True)
#         except Exception as e:
#             print(f"[AI] Engine Warning: Could not load Stockfish ({e}). Advice will be disabled.")

#         # Warmup
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         f, o = segmenter.encode_image(dummy)
#         segmenter.decode_from_features(f, o, "chessboard")
#         segmenter.decode_from_features(f, o, "chess pieces")
#         print("[AI] Ready.", flush=True)

#     except Exception as e:
#         print(f"[AI] Init Failed: {e}", flush=True)
#         traceback.print_exc()
#         return

#     stats_total = []
#     stats_skipped = 0

#     print(f"{'FRAME':<6} | {'ENC':<6} | {'SEG':<6} | {'GRID':<6} | {'PIECE':<6} | {'TOTAL':<6}")

#     try:
#         while True:
#             try:
#                 try: packet = frame_queue.get(timeout=1.0)
#                 except: continue 
#                 if packet is None: break
#                 frame_id, frame = packet
#                 if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
                
#                 t_start = time.perf_counter()

#                 # 1. Encode
#                 vision_features, original_img = segmenter.encode_image(frame)
#                 t_encode = time.perf_counter()

#                 if (t_encode - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
#                     stats_skipped += 1; continue

#                 # 2. Segment Board
#                 mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
#                 is_good, _ = quality_filter.check(mask, score)
#                 if not is_good: stats_skipped += 1; continue

#                 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 if not contours: continue
#                 largest_contour = max(contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
#                 if w < 50 or h < 50: continue

#                 crop_img = original_img[y:y+h, x:x+w].copy()
#                 crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
#                 curr_crop_bbox = (x, y, w, h)
#                 curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                
#                 t_seg = time.perf_counter()
                
#                 # 3. Track / Solve Grid
#                 tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
                
#                 def run_pieces():
#                     ctx = pycuda.autoinit.context
#                     ctx.push()
#                     try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
#                     finally: ctx.pop()
                
#                 pool = ThreadPool(processes=1)
#                 async_pieces = pool.apply_async(run_pieces)
                
#                 success = tracking_success
#                 if not tracking_success:
#                     rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
#                     warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
#                     if success: tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
                
#                 t_grid = time.perf_counter()

#                 if not success:
#                     async_pieces.wait(); pool.close(); pool.join()
#                     stats_skipped += 1; tracker.reset()
#                     # print(f"Frame {frame_id} FAILED: {stats}")
#                     continue

#                 grid_points_local, tile_centers_local = correct_orientation(grid_points_local, tile_centers_local)
#                 piece_masks_list, _, _ = async_pieces.get()
#                 pool.close(); pool.join()

#                 # 4. Process Pieces & Stabilize
#                 grid_points_global = grid_points_local + np.array([x, y])
#                 tile_centers_global = tile_centers_local + np.array([x, y])
                
#                 detected_tiles_map = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)
#                 raw_indices = list(detected_tiles_map.keys())
#                 stable_ascii_map = stabilizer.update(raw_indices)
                
#                 t_end = time.perf_counter()

#                 # --- CALCULATE TIMINGS & LOG ---
#                 ms_total  = (t_end - t_start) * 1000
#                 stats_total.append(ms_total)

#                 # --- GET ADVICE FROM ENGINE (STATE CONTROLLED) ---
#                 current_fen = generate_fen(stable_ascii_map)
#                 best_move_uci = None
                
#                 # NEW LOGIC: Only run engine if the board physically changed!
#                 if engine and current_fen != last_fen_analyzed:
#                     try:
#                         # Append Turn info if missing
#                         fen_to_check = current_fen
#                         if " " not in fen_to_check:
#                              fen_to_check += " w - - 0 1"

#                         board = chess.Board(fen_to_check)
#                         if board.is_valid():
#                             # Analyze
#                             result = engine.play(board, chess.engine.Limit(time=0.1))
#                             if result.move:
#                                 best_move_uci = result.move.uci()
#                                 # Only update our "Memory" if we successfully got a move
#                                 last_fen_analyzed = current_fen
#                                 print(f"[AI] 💡 New Advice Generated: {best_move_uci}")
#                     except Exception as err:
#                         pass
                
#                 # --- SEND RESULT ---
#                 # best_move_uci will be None if FEN didn't change, preventing spam
#                 result_queue.put({
#                     "type": "board_state", 
#                     "fen": current_fen, 
#                     "frame_id": frame_id,
#                     "advice": best_move_uci 
#                 })
                
#                 if SAVE_DEBUG_FRAMES:
#                     vis_img = original_img.copy()
#                     draw_grid_lines(vis_img, grid_points_global)
#                     visualize_pieces(vis_img, detected_tiles_map, tile_centers_global)
#                     cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)

#             except Exception as e:
#                 print(f"[AI] Error: {e}", flush=True)
#                 traceback.print_exc()
#                 tracker.reset()
#                 time.sleep(0.1)

#     finally:
#         if engine:
#             engine.quit()
#             print("[AI] Engine stopped.")
    

    
# def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
#     print("[AI] Worker Started.", flush=True)
    
#     # --- STOCKFISH CONFIGURATION ---
#     # Absolute path based on your previous terminal output
#     STOCKFISH_PATH = "/home/matcha/seenit/Seenitt/core/src/server/stockfish/stockfish-ubuntu-x86-64-avx2"
#     engine = None

#     try:
#         if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         # Initialize Computer Vision components
#         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
#         solver = GridSolver()
#         quality_filter = QualityFilter(margin=5, min_score=0.5)
#         tracker = GridTracker()
#         stabilizer = PositionStabilizer()

#         # --- INITIALIZE ENGINE ---
#         try:
#             # Ensure the file is executable: chmod +x path/to/stockfish
#             engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
#             print("[AI] Stockfish Engine Loaded.", flush=True)
#         except Exception as e:
#             print(f"[AI] Engine Warning: Could not load Stockfish ({e}). Advice will be disabled.")
#             # We don't return here, we let the worker continue without advice if engine fails

#         # Warmup
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         f, o = segmenter.encode_image(dummy)
#         segmenter.decode_from_features(f, o, "chessboard")
#         segmenter.decode_from_features(f, o, "chess pieces")
#         print("[AI] Ready.", flush=True)

#     except Exception as e:
#         print(f"[AI] Init Failed: {e}", flush=True)
#         traceback.print_exc()
#         return

#     stats_total = []
#     stats_skipped = 0

#     print(f"{'FRAME':<6} | {'ENC':<6} | {'SEG':<6} | {'GRID':<6} | {'PIECE':<6} | {'TOTAL':<6}")

#     try: # Try/Finally block to ensure engine cleanup
#         while True:
#             try:
#                 try: packet = frame_queue.get(timeout=1.0)
#                 except: continue 
#                 if packet is None: break
#                 frame_id, frame = packet
#                 if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
                
#                 # --- TIMER START ---
#                 t_start = time.perf_counter()

#                 # 1. Encode
#                 vision_features, original_img = segmenter.encode_image(frame)
                
#                 # --- TIMER CHECKPOINT 1: ENCODE ---
#                 t_encode = time.perf_counter()

#                 if (t_encode - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
#                     stats_skipped += 1; continue

#                 # 2. Segment Board
#                 mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
#                 is_good, _ = quality_filter.check(mask, score)
#                 if not is_good: stats_skipped += 1; continue

#                 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 if not contours: continue
#                 largest_contour = max(contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
#                 if w < 50 or h < 50: continue

#                 crop_img = original_img[y:y+h, x:x+w].copy()
#                 crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
#                 curr_crop_bbox = (x, y, w, h)
#                 curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                
#                 # --- TIMER CHECKPOINT 2: SEGMENTATION ---
#                 t_seg = time.perf_counter()
                
#                 # 3. Track / Solve Grid
#                 tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
                
#                 def run_pieces():
#                     ctx = pycuda.autoinit.context
#                     ctx.push()
#                     try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
#                     finally: ctx.pop()
                
#                 pool = ThreadPool(processes=1)
#                 async_pieces = pool.apply_async(run_pieces)
                
#                 success = tracking_success
#                 if not tracking_success:
#                     rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
#                     warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
#                     if success: tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
                
#                 # --- TIMER CHECKPOINT 3: GRID SOLVER ---
#                 t_grid = time.perf_counter()

#                 if not success:
#                     async_pieces.wait(); pool.close(); pool.join()
#                     stats_skipped += 1; tracker.reset()
#                     print(f"Frame {frame_id} FAILED: {stats}")
#                     continue

#                 grid_points_local, tile_centers_local = correct_orientation(grid_points_local, tile_centers_local)
#                 piece_masks_list, _, _ = async_pieces.get()
#                 pool.close(); pool.join()

#                 # 4. Process Pieces & Stabilize
#                 grid_points_global = grid_points_local + np.array([x, y])
#                 tile_centers_global = tile_centers_local + np.array([x, y])
                
#                 detected_tiles_map = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)
#                 raw_indices = list(detected_tiles_map.keys())
#                 stable_ascii_map = stabilizer.update(raw_indices)
                
#                 # --- TIMER CHECKPOINT 4: END ---
#                 t_end = time.perf_counter()

#                 # --- CALCULATE TIMINGS ---
#                 ms_enc    = (t_encode - t_start) * 1000
#                 ms_seg    = (t_seg - t_encode) * 1000
#                 ms_grid   = (t_grid - t_seg) * 1000
#                 ms_piece  = (t_end - t_grid) * 1000
#                 ms_total  = (t_end - t_start) * 1000
#                 stats_total.append(ms_total)

#                 # --- TERMINAL VISUALIZATION (TIMING + SIDE BY SIDE) ---
#                 print(f"\n[{frame_id:<5}] {ms_enc:.0f}ms enc | {ms_seg:.0f}ms seg | {ms_grid:.0f}ms grid | {ms_piece:.0f}ms piece | = {ms_total:.0f}ms total")
#                 print(f"{'RAW DETECTIONS (Left)':<25} | {'STABILIZED (Right)':<25}")
#                 print("-" * 75)
#                 for row in range(8):
#                     # Build Raw Line
#                     line_raw = ""
#                     for col in range(8):
#                         idx = row * 8 + col
#                         char = 'X' if idx in raw_indices else '.'
#                         line_raw += f"{char} "
                    
#                     # Build Stable Line
#                     line_stable = ""
#                     for col in range(8):
#                         idx = row * 8 + col
#                         char = stable_ascii_map.get(idx, '.')
#                         line_stable += f"{char} "
                    
#                     print(f"Row {8-row}       | {line_raw:<25} | {line_stable:<25}")
#                 print("-" * 75)

#                 # --- GET ADVICE FROM ENGINE ---
#                 current_fen = generate_fen(stable_ascii_map)
#                 best_move_uci = ""
                
#                 if engine:
#                     try:
#                         # Add default turn info if missing (usually assumes White to move if not specified)
#                         # or you can append " w - - 0 1" if your FEN generator doesn't do it.
#                         fen_to_check = current_fen
#                         if " " not in fen_to_check:
#                              fen_to_check += " w - - 0 1"

#                         board = chess.Board(fen_to_check)
#                         if board.is_valid():
#                             # Analyze for 0.1s max to keep video smooth
#                             result = engine.play(board, chess.engine.Limit(time=0.1))
#                             if result.move:
#                                 best_move_uci = result.move.uci()
#                     except Exception as err:
#                         # Engine errors (e.g., illegal position) shouldn't crash the worker
#                         pass

#                 # --- SEND RESULT ---
#                 result_queue.put({
#                     "type": "board_state", 
#                     "fen": current_fen, 
#                     "frame_id": frame_id,
#                     "advice": best_move_uci
#                 })
                
#                 if SAVE_DEBUG_FRAMES:
#                     vis_img = original_img.copy()
#                     draw_grid_lines(vis_img, grid_points_global)
#                     visualize_pieces(vis_img, detected_tiles_map, tile_centers_global)
#                     cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)

#             except Exception as e:
#                 print(f"[AI] Error: {e}", flush=True)
#                 traceback.print_exc()
#                 tracker.reset()
#                 time.sleep(0.1)

#     finally:
#         if engine:
#             engine.quit()
#             print("[AI] Engine stopped.")








# import os
# import time
# import traceback
# import cv2
# import numpy as np
# import pycuda.autoinit
# from multiprocessing.pool import ThreadPool

# from src.segmentation import ChessboardSegmenter
# from src.grid_solver import GridSolver
# from src.filter import QualityFilter
# from src.grid_tracker import GridTracker
# from src.piece_processor import process_pieces
# from src.visualizer import draw_grid_lines, visualize_pieces
# from src.server.config import TOTAL_TIME_LIMIT_MS, SAVE_DEBUG_FRAMES, OUTPUT_DIR
# from src.server.utils import print_ascii_board
# from src.stabilizer import PositionStabilizer 

# # --- HELPER: ORIENTATION CORRECTION ---
# def correct_orientation(grid_points, tile_centers):
#     g = grid_points.reshape(9, 9, 2)
#     c = tile_centers.reshape(8, 8, 2)
#     best_dist = float('inf')
#     best_k = 0
#     for k in range(4):
#         rotated_g = np.rot90(g, k)
#         p0 = rotated_g[0, 0]
#         dist = (p0[0] ** 2) + (p0[1] ** 2)
#         if dist < best_dist:
#             best_dist = dist
#             best_k = k
#     g = np.rot90(g, best_k)
#     c = np.rot90(c, best_k)
#     p0, p1 = g[0, 0], g[0, 1]
#     dx, dy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
#     if dy > dx:
#         g = np.transpose(g, (1, 0, 2))
#         c = np.transpose(c, (1, 0, 2))
#     return np.ascontiguousarray(g.reshape(-1, 2)), np.ascontiguousarray(c.reshape(-1, 2))

# # --- HELPER: FEN GENERATOR ---
# def generate_fen(ascii_map):
#     fen = ""
#     for row in range(8):
#         empty = 0
#         for col in range(8):
#             idx = row * 8 + col
#             char = ascii_map.get(idx, '.')
#             if char == '.':
#                 empty += 1
#             else:
#                 if empty > 0:
#                     fen += str(empty)
#                     empty = 0
#                 fen += char
#         if empty > 0: fen += str(empty)
#         if row < 7: fen += "/"
#     return fen

# def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
#     print("[AI] Worker Started.", flush=True)
#     try:
#         if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
#         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
#         solver = GridSolver()
#         quality_filter = QualityFilter(margin=5, min_score=0.5)
#         tracker = GridTracker()
#         stabilizer = PositionStabilizer()

#         # Warmup
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         f, o = segmenter.encode_image(dummy)
#         segmenter.decode_from_features(f, o, "chessboard")
#         segmenter.decode_from_features(f, o, "chess pieces")
#         print("[AI] Ready.", flush=True)
#     except Exception as e:
#         print(f"[AI] Init Failed: {e}", flush=True)
#         traceback.print_exc()
#         return

#     stats_total = []
#     stats_skipped = 0

#     # Optional Header
#     print(f"{'FRAME':<6} | {'ENC':<6} | {'SEG':<6} | {'GRID':<6} | {'PIECE':<6} | {'TOTAL':<6}")

#     while True:
#         try:
#             try: packet = frame_queue.get(timeout=1.0)
#             except: continue 
#             if packet is None: break
#             frame_id, frame = packet
#             if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
            
#             # --- TIMER START ---
#             t_start = time.perf_counter()

#             # 1. Encode
#             vision_features, original_img = segmenter.encode_image(frame)
            
#             # --- TIMER CHECKPOINT 1: ENCODE ---
#             t_encode = time.perf_counter()

#             if (t_encode - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
#                 stats_skipped += 1; continue

#             # 2. Segment Board
#             mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
#             is_good, _ = quality_filter.check(mask, score)
#             if not is_good: stats_skipped += 1; continue

#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             if not contours: continue
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             if w < 50 or h < 50: continue

#             crop_img = original_img[y:y+h, x:x+w].copy()
#             crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
#             curr_crop_bbox = (x, y, w, h)
#             curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
#             # --- TIMER CHECKPOINT 2: SEGMENTATION ---
#             t_seg = time.perf_counter()
            
#             # 3. Track / Solve Grid
#             tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
            
#             def run_pieces():
#                 ctx = pycuda.autoinit.context
#                 ctx.push()
#                 try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
#                 finally: ctx.pop()
            
#             pool = ThreadPool(processes=1)
#             async_pieces = pool.apply_async(run_pieces)
            
#             success = tracking_success
#             if not tracking_success:
#                 rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
#                 warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
#                 if success: tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
            
#             # --- TIMER CHECKPOINT 3: GRID SOLVER ---
#             t_grid = time.perf_counter()

#             if not success:
#                 async_pieces.wait(); pool.close(); pool.join()
#                 stats_skipped += 1; tracker.reset()
#                 print(f"Frame {frame_id} FAILED: {stats}")
#                 continue

#             grid_points_local, tile_centers_local = correct_orientation(grid_points_local, tile_centers_local)
#             piece_masks_list, _, _ = async_pieces.get()
#             pool.close(); pool.join()

#             # 4. Process Pieces & Stabilize
#             grid_points_global = grid_points_local + np.array([x, y])
#             tile_centers_global = tile_centers_local + np.array([x, y])
            
#             detected_tiles_map = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)
#             raw_indices = list(detected_tiles_map.keys())
#             stable_ascii_map = stabilizer.update(raw_indices)
            
#             # --- TIMER CHECKPOINT 4: END ---
#             t_end = time.perf_counter()

#             # --- CALCULATE TIMINGS ---
#             ms_enc   = (t_encode - t_start) * 1000
#             ms_seg   = (t_seg - t_encode) * 1000
#             ms_grid  = (t_grid - t_seg) * 1000
#             ms_piece = (t_end - t_grid) * 1000
#             ms_total = (t_end - t_start) * 1000
#             stats_total.append(ms_total)

#             # --- TERMINAL VISUALIZATION (TIMING + SIDE BY SIDE) ---
#             print(f"\n[{frame_id:<5}] {ms_enc:.0f}ms enc | {ms_seg:.0f}ms seg | {ms_grid:.0f}ms grid | {ms_piece:.0f}ms piece | = {ms_total:.0f}ms total")
#             print(f"{'RAW DETECTIONS (Left)':<25} | {'STABILIZED (Right)':<25}")
#             print("-" * 75)
#             for row in range(8):
#                 # Build Raw Line
#                 line_raw = ""
#                 for col in range(8):
#                     idx = row * 8 + col
#                     # Show 'X' if a piece was detected in this specific frame
#                     char = 'X' if idx in raw_indices else '.'
#                     line_raw += f"{char} "
                
#                 # Build Stable Line
#                 line_stable = ""
#                 for col in range(8):
#                     idx = row * 8 + col
#                     # Show the stable piece character
#                     char = stable_ascii_map.get(idx, '.')
#                     line_stable += f"{char} "
                
#                 print(f"Row {8-row}      | {line_raw:<25} | {line_stable:<25}")
#             print("-" * 75)

#             # --- SEND FEN TO SERVER ---
#             current_fen = generate_fen(stable_ascii_map)
#             result_queue.put({"type": "board_state", "fen": current_fen, "frame_id": frame_id})
            
#             if SAVE_DEBUG_FRAMES:
#                 vis_img = original_img.copy()
#                 draw_grid_lines(vis_img, grid_points_global)
#                 visualize_pieces(vis_img, detected_tiles_map, tile_centers_global)
#                 cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)

#         except Exception as e:
#             print(f"[AI] Error: {e}", flush=True)
#             traceback.print_exc()
#             tracker.reset()
#             time.sleep(0.1)







# import os
# import time
# import traceback
# import cv2
# import numpy as np
# import pycuda.autoinit
# from multiprocessing.pool import ThreadPool

# from src.segmentation import ChessboardSegmenter
# from src.grid_solver import GridSolver
# from src.filter import QualityFilter
# from src.grid_tracker import GridTracker
# from src.piece_processor import process_pieces
# from src.visualizer import draw_grid_lines, visualize_pieces
# from src.server.config import TOTAL_TIME_LIMIT_MS, SAVE_DEBUG_FRAMES, OUTPUT_DIR
# from src.server.utils import print_ascii_board
# from src.stabilizer import PositionStabilizer 

# # --- HELPER: ORIENTATION CORRECTION ---
# def correct_orientation(grid_points, tile_centers):
#     g = grid_points.reshape(9, 9, 2)
#     c = tile_centers.reshape(8, 8, 2)
#     best_dist = float('inf')
#     best_k = 0
#     for k in range(4):
#         rotated_g = np.rot90(g, k)
#         p0 = rotated_g[0, 0]
#         dist = (p0[0] ** 2) + (p0[1] ** 2)
#         if dist < best_dist:
#             best_dist = dist
#             best_k = k
#     g = np.rot90(g, best_k)
#     c = np.rot90(c, best_k)
#     p0, p1 = g[0, 0], g[0, 1]
#     dx, dy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
#     if dy > dx:
#         g = np.transpose(g, (1, 0, 2))
#         c = np.transpose(c, (1, 0, 2))
#     return np.ascontiguousarray(g.reshape(-1, 2)), np.ascontiguousarray(c.reshape(-1, 2))

# # --- HELPER: FEN GENERATOR ---
# def generate_fen(ascii_map):
#     fen = ""
#     for row in range(8):
#         empty = 0
#         for col in range(8):
#             idx = row * 8 + col
#             char = ascii_map.get(idx, '.')
#             if char == '.':
#                 empty += 1
#             else:
#                 if empty > 0:
#                     fen += str(empty)
#                     empty = 0
#                 fen += char
#         if empty > 0: fen += str(empty)
#         if row < 7: fen += "/"
#     return fen

# def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
#     print("[AI] Worker Started.", flush=True)
#     try:
#         if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
#         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
#         solver = GridSolver()
#         quality_filter = QualityFilter(margin=5, min_score=0.5)
#         tracker = GridTracker()
#         stabilizer = PositionStabilizer()

#         # Warmup
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         f, o = segmenter.encode_image(dummy)
#         segmenter.decode_from_features(f, o, "chessboard")
#         segmenter.decode_from_features(f, o, "chess pieces")
#         print("[AI] Ready.", flush=True)
#     except Exception as e:
#         print(f"[AI] Init Failed: {e}", flush=True)
#         traceback.print_exc()
#         return

#     stats_total = []
#     stats_skipped = 0

#     while True:
#         try:
#             try: packet = frame_queue.get(timeout=1.0)
#             except: continue 
#             if packet is None: break
#             frame_id, frame = packet
#             if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
#             t_start = time.perf_counter()

#             # 1. Encode
#             vision_features, original_img = segmenter.encode_image(frame)
#             if (time.perf_counter() - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
#                 stats_skipped += 1; continue

#             # 2. Segment Board
#             mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
#             is_good, _ = quality_filter.check(mask, score)
#             if not is_good: stats_skipped += 1; continue

#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             if not contours: continue
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             if w < 50 or h < 50: continue

#             crop_img = original_img[y:y+h, x:x+w].copy()
#             crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
#             curr_crop_bbox = (x, y, w, h)
#             curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
#             # 3. Track / Solve Grid
#             tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
            
#             def run_pieces():
#                 ctx = pycuda.autoinit.context
#                 ctx.push()
#                 try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
#                 finally: ctx.pop()
            
#             pool = ThreadPool(processes=1)
#             async_pieces = pool.apply_async(run_pieces)
            
#             success = tracking_success
#             if not tracking_success:
#                 rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
#                 warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
#                 if success: tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
            
#             if not success:
#                 async_pieces.wait(); pool.close(); pool.join()
#                 stats_skipped += 1; tracker.reset()
#                 print(f"Frame {frame_id} FAILED: {stats}")
#                 continue

#             grid_points_local, tile_centers_local = correct_orientation(grid_points_local, tile_centers_local)
#             piece_masks_list, _, _ = async_pieces.get()
#             pool.close(); pool.join()

#             # 4. Process Pieces & Stabilize
#             grid_points_global = grid_points_local + np.array([x, y])
#             tile_centers_global = tile_centers_local + np.array([x, y])
            
#             detected_tiles_map = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)
#             raw_indices = list(detected_tiles_map.keys())
#             stable_ascii_map = stabilizer.update(raw_indices)
            
#             # --- TERMINAL VISUALIZATION (SIDE BY SIDE) ---
#             print(f"\nFrame {frame_id:<5} | {'RAW DETECTIONS (Left)':<25} | {'STABILIZED (Right)':<25}")
#             print("-" * 75)
#             for row in range(8):
#                 # Build Raw Line
#                 line_raw = ""
#                 for col in range(8):
#                     idx = row * 8 + col
#                     # Show 'X' if a piece was detected in this specific frame
#                     char = 'X' if idx in raw_indices else '.'
#                     line_raw += f"{char} "
                
#                 # Build Stable Line
#                 line_stable = ""
#                 for col in range(8):
#                     idx = row * 8 + col
#                     # Show the stable piece character
#                     char = stable_ascii_map.get(idx, '.')
#                     line_stable += f"{char} "
                
#                 print(f"Row {8-row}     | {line_raw:<25} | {line_stable:<25}")
#             print("-" * 75)

#             # --- SEND FEN TO SERVER ---
#             current_fen = generate_fen(stable_ascii_map)
#             result_queue.put({"type": "board_state", "fen": current_fen, "frame_id": frame_id})
            
#             # Logging
#             total_ms = (time.perf_counter() - t_start) * 1000.0
#             stats_total.append(total_ms)
            
#             if SAVE_DEBUG_FRAMES:
#                 vis_img = original_img.copy()
#                 draw_grid_lines(vis_img, grid_points_global)
#                 visualize_pieces(vis_img, detected_tiles_map, tile_centers_global)
#                 cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)

#         except Exception as e:
#             print(f"[AI] Error: {e}", flush=True)
#             traceback.print_exc()
#             tracker.reset()
#             time.sleep(0.1)












# import os
# import time
# import traceback
# import cv2
# import numpy as np
# import pycuda.autoinit
# from multiprocessing.pool import ThreadPool

# # Import Engines
# from src.segmentation import ChessboardSegmenter
# from src.grid_solver import GridSolver
# from src.filter import QualityFilter
# from src.grid_tracker import GridTracker
# from src.piece_processor import process_pieces
# from src.visualizer import draw_grid_lines, visualize_pieces

# # Import Server Utils
# from src.server.config import TOTAL_TIME_LIMIT_MS, SAVE_DEBUG_FRAMES, OUTPUT_DIR
# from src.server.utils import print_ascii_board

# # Stabilization
# from src.stabilizer import PositionStabilizer 


# # --- HELPER: ORIENTATION CORRECTION ---
# def correct_orientation(grid_points, tile_centers):
#     """
#     1. Finds the rotation that puts Index [0,0] closest to image origin (0,0).
#     2. Checks if the grid is "transposed" (reading down instead of right) and fixes it.
#     """
#     g = grid_points.reshape(9, 9, 2)
#     c = tile_centers.reshape(8, 8, 2)
    
#     # STEP 1: Fix Rotation (Find the Top-Left Corner)
#     # We rotate 90 degrees until g[0,0] is the corner closest to pixel (0,0)
#     best_dist = float('inf')
#     best_k = 0
    
#     for k in range(4):
#         # Rotate grid k times
#         rotated_g = np.rot90(g, k)
#         p0 = rotated_g[0, 0] # The "first" point in this rotation
        
#         # Distance to Top-Left image corner (0,0)
#         dist = (p0[0] ** 2) + (p0[1] ** 2)
        
#         if dist < best_dist:
#             best_dist = dist
#             best_k = k

#     # Apply the best rotation found
#     g = np.rot90(g, best_k)
#     c = np.rot90(c, best_k)

#     # STEP 2: Fix Transpose (Ensure Left-to-Right Reading)
#     # Now that g[0,0] is definitely Top-Left, we check the next point g[0,1].
#     # It SHOULD be to the Right (+X). If it's Down (+Y), the grid is transposed.
    
#     p0 = g[0, 0]
#     p1 = g[0, 1]
#     dx = abs(p1[0] - p0[0])
#     dy = abs(p1[1] - p0[1])

#     # If change in Y is bigger than change in X, we are reading Down (Wrong).
#     # Swap the axes (Transpose) to force Left-to-Right reading.
#     if dy > dx:
#         g = np.transpose(g, (1, 0, 2))
#         c = np.transpose(c, (1, 0, 2))

#     return np.ascontiguousarray(g.reshape(-1, 2)), np.ascontiguousarray(c.reshape(-1, 2))



# def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
#     print("[AI] Worker Started (Chess Engine Mode).", flush=True)
#     try:
#         if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
#         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
#         solver = GridSolver()
#         quality_filter = QualityFilter(margin=5, min_score=0.5)
#         tracker = GridTracker()

#         # Warmup
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         f, o = segmenter.encode_image(dummy)
#         segmenter.decode_from_features(f, o, "chessboard")
#         segmenter.decode_from_features(f, o, "chess pieces")
#         print("[AI] Ready.", flush=True)
#     except Exception as e:
#         print(f"[AI] Init Failed: {e}", flush=True)
#         traceback.print_exc()
#         return

#     stats_total = []
#     stats_skipped = 0

#     print(f"{'FRAME':<10} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS':<8} | {'REASON'}")
#     print("-" * 100)

#     # Initialize the Chess Engine Stabilizer
#     stabilizer = PositionStabilizer()
    
#     while True:
#         try:
#             try: packet = frame_queue.get(timeout=1.0)
#             except: continue 
#             if packet is None: break
#             frame_id, frame = packet
#             if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
#             t_start = time.perf_counter()

#             # 1. Encode
#             vision_features, original_img = segmenter.encode_image(frame)
#             t_encode = time.perf_counter()
#             if (t_encode - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
#                 stats_skipped += 1; continue

#             # 2. Segment Board
#             mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
#             t_seg = time.perf_counter()
            
#             is_good, _ = quality_filter.check(mask, score)
#             if not is_good: stats_skipped += 1; continue

#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             if not contours: continue
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             if w < 50 or h < 50: continue

#             crop_img = original_img[y:y+h, x:x+w].copy()
#             crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
#             curr_crop_bbox = (x, y, w, h)
#             curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
#             # 3. Track / Solve Grid
#             tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
            
#             def run_pieces():
#                 ctx = pycuda.autoinit.context
#                 ctx.push()
#                 try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
#                 finally: ctx.pop()
            
#             pool = ThreadPool(processes=1)
#             async_pieces = pool.apply_async(run_pieces)
            
#             t_parallel = time.perf_counter()
#             success = tracking_success
            
#             if not tracking_success:
#                 rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
#                 warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
#                 if success:
#                     tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
            
#             t_grid = time.perf_counter()
            
#             if not success:
#                 async_pieces.wait(); pool.close(); pool.join()
#                 stats_skipped += 1; tracker.reset()
#                 error_reason = str(stats) if stats else "Unknown Error"
#                 print(f"Frame {frame_id:<10} | {(t_seg-t_start)*1000:<8.1f} | {(t_grid-t_parallel)*1000:<8.1f} | -        | -        | FAILED   | {error_reason}")
#                 continue

#             # === FORCE ORIENTATION ===
#             grid_points_local, tile_centers_local = correct_orientation(grid_points_local, tile_centers_local)
#             # =================================================
            
#             piece_masks_list, _, _ = async_pieces.get()
#             pool.close(); pool.join()
#             t_pieces = time.perf_counter()

#             # --- PROCESS PIECES ---
#             grid_points_global = grid_points_local + np.array([x, y])
#             tile_centers_global = tile_centers_local + np.array([x, y])
            
#             detected_tiles_map = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)

#             # 1. RAW detected indices
#             raw_indices = list(detected_tiles_map.keys())
            
#             # 2. Stabilize
#             stable_ascii_map = stabilizer.update(raw_indices)
            
#             # --- SIDE-BY-SIDE ASCII DISPLAY ---
#             print(f"\nFrame {frame_id} Comparison:")
#             print(f"{'RAW DETECTIONS (Camera)':<40} | {'STABILIZED (Chess Engine)':<40}")
#             print("-" * 85)
#             for row in range(8):
#                 # Build Raw Line
#                 line_raw = f"{8-row} | "
#                 for col in range(8):
#                     idx = row * 8 + col
#                     char = 'X' if idx in raw_indices else '.'
#                     line_raw += f"{char} "
#                 line_raw += "|"

#                 # Build Stable Line
#                 line_stable = f"{8-row} | "
#                 for col in range(8):
#                     idx = row * 8 + col
#                     # Fetch symbol (R, r, P, etc) directly from the map
#                     # If the key is missing, it means Empty Square (.)
#                     char = stable_ascii_map.get(idx, '.')
#                     line_stable += f"{char} "
#                 line_stable += "|"
                
#                 print(f"{line_raw:<40} | {line_stable:<40}")
#             print(f"{'    a b c d e f g h':<40} | {'    a b c d e f g h':<40}")
#             print("-" * 85)
            
#             # --- VISUAL DEBUG ---
#             if SAVE_DEBUG_FRAMES:
#                 vis_img = original_img.copy()
#                 draw_grid_lines(vis_img, grid_points_global)
#                 visualize_pieces(vis_img, detected_tiles_map, tile_centers_global)
#                 cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)
            
#             total_ms = (t_pieces - t_start) * 1000.0
#             stats_total.append(total_ms)
            
#             print(f"Frame {frame_id:<10} | {(t_seg-t_start)*1000:<8.1f} | {(t_grid-t_parallel)*1000:<8.1f} | {(t_pieces-t_parallel)*1000 - (t_grid-t_parallel)*1000:<8.1f} | {total_ms:<8.1f} | OK")
            
#             if frame_id % 100 == 0 and len(stats_total) > 0:
#                 result_queue.put_nowait(f"Status: {len(stats_total)} OK, {stats_skipped} skipped, {sum(stats_total)/len(stats_total):.1f}ms avg")

#         except Exception as e:
#             print(f"[AI] Error: {e}", flush=True)
#             traceback.print_exc()
#             tracker.reset()
#             time.sleep(0.1)


# import os
# import time
# import traceback
# import cv2
# import numpy as np
# import pycuda.autoinit
# from multiprocessing.pool import ThreadPool

# # Import Engines
# from src.segmentation import ChessboardSegmenter
# from src.grid_solver import GridSolver
# from src.filter import QualityFilter
# from src.grid_tracker import GridTracker
# from src.piece_processor import process_pieces
# from src.visualizer import draw_grid_lines, visualize_pieces

# # Import Server Utils
# from src.server.config import TOTAL_TIME_LIMIT_MS, SAVE_DEBUG_FRAMES, OUTPUT_DIR
# from src.server.utils import print_ascii_board

# # Import chess logic code
# from src.server.chess_state import ChessboardState
# from src.server.chess_mem import BoardTracker
# from src.server.PieceClassifier import PieceClassifier

# # MAPPING: Standard ImageFolder sorts ASCII: Uppercase (White) first, then Lowercase (Black).
# # Ensure this matches your training dataset class order.
# IDX_TO_FEN = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r']

# def recalculate_tile_centers(grid_points):
#     g9 = grid_points.reshape(9, 9, 2)
#     centers = []
#     for r in range(8):
#         for c in range(8):
#             p = (g9[r,c] + g9[r,c+1] + g9[r+1,c] + g9[r+1,c+1]) / 4.0
#             centers.append(p)
#     return np.array(centers, dtype=np.float32)

# def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
#     print("[AI] Worker Started.", flush=True)
#     try:
#         if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
#         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
#         solver = GridSolver()
#         quality_filter = QualityFilter(margin=1, min_score=0.5, max_border_contact_ratio=0.05)
        
#         # Load the classifier
#         # Ensure 'best_0.0555.pt' is in the root or provide absolute path
#         piece_classifier = PieceClassifier(load_model_path="src/server/best_0.0555.pt")
        
#         tracker = GridTracker()

#         # Warmup
#         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
#         f, o = segmenter.encode_image(dummy)
#         segmenter.decode_from_features(f, o, "chessboard")
#         segmenter.decode_from_features(f, o, "chess pieces")
#         print("[AI] Ready.", flush=True)
#     except Exception as e:
#         print(f"[AI] Init Failed: {e}", flush=True)
#         traceback.print_exc()
#         return

#     stats_total = []
#     stats_skipped = 0

#     print(f"{'FRAME':<10} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS'}")
#     print("-" * 85)

#     # init chess logic code
#     board = ChessboardState(inc=0.1, dec=0.05)
#     board.init_standard()
#     board_mem = BoardTracker(board=board,
#                              max_changed_tiles=2,
#                              diff_conf_thr=0.6,
#                              min_keep=0.35,
#                             )
     
#     while True:
#         try:
#             try: packet = frame_queue.get(timeout=1.0)
#             except: continue 
#             if packet is None: break
#             frame_id, frame = packet
#             if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
#             t_start = time.perf_counter()

#             # 1. Encode
#             vision_features, original_img = segmenter.encode_image(frame)
#             t_encode = time.perf_counter()
#             if (t_encode - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
#                 stats_skipped += 1; continue

#             # 2. Segment Board
#             mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
#             t_seg = time.perf_counter()
            
#             is_good, _ = quality_filter.check(mask, score)
#             if not is_good: stats_skipped += 1; continue

#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             if not contours: continue
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             if w < 50 or h < 50: continue

#             crop_img = original_img[y:y+h, x:x+w].copy()
#             crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
#             curr_crop_bbox = (x, y, w, h)
#             curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
#             # 3. Track / Solve Grid
#             tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
            
#             def run_pieces():
#                 ctx = pycuda.autoinit.context
#                 ctx.push()
#                 try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
#                 finally: ctx.pop()
            
#             pool = ThreadPool(processes=1)
#             async_pieces = pool.apply_async(run_pieces)
            
#             t_parallel = time.perf_counter()
#             success = tracking_success
            
#             if not tracking_success:
#                 rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
#                 warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
#                 if success:
#                     tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
            
#             t_grid = time.perf_counter()
#             if not success:
#                 async_pieces.wait(); pool.close(); pool.join()
#                 stats_skipped += 1; tracker.reset()
#                 print(f"Frame {frame_id:<10} | {(t_seg-t_start)*1000:<8.1f} | {(t_grid-t_parallel)*1000:<8.1f} | - | - | FAILED")
#                 continue

#             piece_masks_list, _, _ = async_pieces.get()
#             pool.close(); pool.join()
#             t_pieces = time.perf_counter()

#             # --- PROCESS PIECES ---
#             grid_points_global = grid_points_local + np.array([x, y])
#             tile_centers_global = tile_centers_local + np.array([x, y])
            
#             # Identify OCCUPIED tiles using the segmentation masks
#             # (tile_to_piece here just contains "True" or dummy values, we need to classify them)
#             detected_tiles_map = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)

#             # --- CLASSIFICATION & STABILIZATION ---
#             tile_cands = {}
            
#             # Reshape grid to 9x9 for corner extraction
#             g9 = grid_points_global.reshape(9, 9, 2)

#             for t_idx in detected_tiles_map.keys():
#                 # 1. Extract Crop for this tile
#                 r, c = divmod(t_idx, 8)
                
#                 # Get 4 corners of the tile
#                 src_pts = np.array([
#                     g9[r, c],     g9[r, c+1],
#                     g9[r+1, c],   g9[r+1, c+1]
#                 ], dtype=np.float32)

#                 # Destination square (128x128 is good for ResNet)
#                 dst_size = 128
#                 dst_pts = np.array([
#                     [0, 0], [dst_size, 0],
#                     [0, dst_size], [dst_size, dst_size]
#                 ], dtype=np.float32)

#                 # Warp
#                 M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#                 tile_crop = cv2.warpPerspective(original_img, M, (dst_size, dst_size))

#                 # 2. Run Classifier (Get Top 3 predictions)
#                 top_indices, top_probs = piece_classifier.predict_topk(tile_crop, k=3)

#                 # 3. Format for BoardTracker
#                 # It expects a list of (PieceChar, Confidence) tuples
#                 cands_list = []
#                 for i in range(len(top_indices)):
#                     p_idx = top_indices[i]
#                     p_conf = float(top_probs[i])
                    
#                     if p_idx < len(IDX_TO_FEN):
#                         fen_char = IDX_TO_FEN[p_idx]
#                         cands_list.append((fen_char, p_conf))
                
#                 tile_cands[t_idx] = cands_list

#             # 4. Update Board History (Stabilization)
#             is_stable, _, _ = board_mem.step(tile_cands, frame_id)

#             # 5. Overwrite display with Stable State
#             tile_to_piece = {}
#             for i in range(64):
#                 stable_p = board_mem.board.get_piece(i)
#                 if stable_p:
#                     tile_to_piece[i] = stable_p
            
#             # --- DISPLAY ---
#             if SAVE_DEBUG_FRAMES:
#                 vis_img = original_img.copy()
#                 draw_grid_lines(vis_img, grid_points_global)
#                 visualize_pieces(vis_img, tile_to_piece, tile_centers_global)
#                 cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)
            
#             total_ms = (t_pieces - t_start) * 1000.0
#             stats_total.append(total_ms)
            
#             print(f"Frame {frame_id:<10} | {(t_seg-t_start)*1000:<8.1f} | {(t_grid-t_parallel)*1000:<8.1f} | {(t_pieces-t_parallel)*1000 - (t_grid-t_parallel)*1000:<8.1f} | {total_ms:<8.1f} | OK")
            
#             if tile_to_piece:
#                 print_ascii_board(tile_to_piece, frame_id)

#             if frame_id % 100 == 0 and len(stats_total) > 0:
#                 result_queue.put_nowait(f"Status: {len(stats_total)} OK, {stats_skipped} skipped, {sum(stats_total)/len(stats_total):.1f}ms avg")

#         except Exception as e:
#             print(f"[AI] Error: {e}", flush=True)
#             traceback.print_exc()
#             tracker.reset()
#             time.sleep(0.1)




# # # import os
# # # import time
# # # import traceback
# # # import cv2
# # # import numpy as np
# # # import pycuda.autoinit
# # # from multiprocessing.pool import ThreadPool

# # # # Import Engines
# # # from src.segmentation import ChessboardSegmenter
# # # from src.grid_solver import GridSolver
# # # from src.filter import QualityFilter
# # # from src.grid_tracker import GridTracker
# # # from src.piece_processor import process_pieces
# # # from src.visualizer import draw_grid_lines, visualize_pieces

# # # # Import Server Utils
# # # from src.server.config import TOTAL_TIME_LIMIT_MS, SAVE_DEBUG_FRAMES, OUTPUT_DIR
# # # from src.server.utils import print_ascii_board

# # # # Import chess logic code
# # # from src.server.chess_state import ChessboardState
# # # from src.server.chess_mem import BoardTracker
# # # from src.server.PieceClassifier import PieceClassifier


# # # def recalculate_tile_centers(grid_points):
# # #     g9 = grid_points.reshape(9, 9, 2)
# # #     centers = []
# # #     for r in range(8):
# # #         for c in range(8):
# # #             p = (g9[r,c] + g9[r,c+1] + g9[r+1,c] + g9[r+1,c+1]) / 4.0
# # #             centers.append(p)
# # #     return np.array(centers, dtype=np.float32)

# # # def ai_worker_process(frame_queue, result_queue, assets_dir="./assets"):
# # #     print("[AI] Worker Started.", flush=True)
# # #     try:
# # #         if SAVE_DEBUG_FRAMES: os.makedirs(OUTPUT_DIR, exist_ok=True)
# # #         segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
# # #         solver = GridSolver()
# # #         quality_filter = QualityFilter(margin=1, min_score=0.5, max_border_contact_ratio=0.05)

# # #         piece_classifier = PieceClassifier(load_model_path="src/server/best_0.0555.pt")
        
# # #         tracker = GridTracker()

# # #         # Warmup
# # #         dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
# # #         f, o = segmenter.encode_image(dummy)
# # #         segmenter.decode_from_features(f, o, "chessboard")
# # #         segmenter.decode_from_features(f, o, "chess pieces")
# # #         print("[AI] Ready.", flush=True)
# # #     except Exception as e:
# # #         print(f"[AI] Init Failed: {e}", flush=True); traceback.print_exc(); return

# # #     stats_total = []
# # #     stats_skipped = 0

# # #     print(f"{'FRAME':<10} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS'}")
# # #     print("-" * 85)



# # #     # init chess logic code
# # #     board = ChessboardState(inc=0.1, dec=0.05)
# # #     board.init_standard()
# # #     board_mem = BoardTracker(board=board,
# # #                              max_changed_tiles=2,
# # #                              diff_conf_thr=0.6,
# # #                              min_keep=0.35,
# # #                             )
    
# # #     while True:
# # #         try:
# # #             try: packet = frame_queue.get(timeout=1.0)
# # #             except: continue 
# # #             if packet is None: break
# # #             frame_id, frame = packet
# # #             if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
# # #             t_start = time.perf_counter()

# # #             # 1. Encode
# # #             vision_features, original_img = segmenter.encode_image(frame)
# # #             t_encode = time.perf_counter()
# # #             if (t_encode - t_start)*1000 > TOTAL_TIME_LIMIT_MS:
# # #                 stats_skipped += 1; continue

# # #             # 2. Segment Board
# # #             mask, _, score = segmenter.decode_from_features(vision_features, original_img, "chessboard")
# # #             t_seg = time.perf_counter()
            
# # #             is_good, _ = quality_filter.check(mask, score)
# # #             if not is_good: stats_skipped += 1; continue

# # #             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # #             if not contours: continue
# # #             largest_contour = max(contours, key=cv2.contourArea)
# # #             x, y, w, h = cv2.boundingRect(largest_contour)
# # #             if w < 50 or h < 50: continue

# # #             crop_img = original_img[y:y+h, x:x+w].copy()
# # #             crop_mask = np.ascontiguousarray(mask[y:y+h, x:x+w])
# # #             curr_crop_bbox = (x, y, w, h)
# # #             curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
# # #             # 3. Track / Solve Grid
# # #             tracking_success, grid_points_local, tile_centers_local, stats = tracker.try_track(curr_gray, curr_crop_bbox)
            
# # #             def run_pieces():
# # #                 ctx = pycuda.autoinit.context
# # #                 ctx.push()
# # #                 try: return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
# # #                 finally: ctx.pop()
            
# # #             pool = ThreadPool(processes=1)
# # #             async_pieces = pool.apply_async(run_pieces)
            
# # #             t_parallel = time.perf_counter()
# # #             success = tracking_success
            
# # #             if not tracking_success:
# # #                 rem_ms = TOTAL_TIME_LIMIT_MS - ((time.perf_counter() - t_start) * 1000)
# # #                 warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(crop_img, crop_mask, time_limit_ms=rem_ms)
# # #                 if success:
# # #                     # History alignment removed here (Raw Grid Solver output used directly)
# # #                     tracker.update(curr_gray, grid_points_local, curr_crop_bbox)
            
# # #             t_grid = time.perf_counter()
# # #             if not success:
# # #                 async_pieces.wait(); pool.close(); pool.join()
# # #                 stats_skipped += 1; tracker.reset()
# # #                 print(f"Frame {frame_id:<10} | {(t_seg-t_start)*1000:<8.1f} | {(t_grid-t_parallel)*1000:<8.1f} | - | - | FAILED")
# # #                 continue

# # #             piece_masks_list, _, _ = async_pieces.get()
# # #             pool.close(); pool.join()
# # #             t_pieces = time.perf_counter()

# # #             # --- PROCESS PIECES ---
# # #             grid_points_global = grid_points_local + np.array([x, y])
# # #             tile_centers_global = tile_centers_local + np.array([x, y])
            
# # #             # Map pieces to the current grid (Raw orientation)
# # #             tile_to_piece = process_pieces(piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global)



# # #             # STABILIZATION CODE
# # #             tile_cands = {}
# # #             for t_idx, val in tile_to_piece.items():
# # #                 # process_pieces might return (piece, conf, extra_data...) or lists of them
# # #                 # We need to strictly extract (piece, conf) for the BoardTracker
                
# # #                 # Helper to clean a single candidate tuple
# # #                 def clean_tuple(tup):
# # #                     return (tup[0], float(tup[1])) # strictly (str, float)

# # #                 if isinstance(val, list):
# # #                     # If it's already a list of candidates, clean each one
# # #                     tile_cands[t_idx] = [clean_tuple(v) for v in val]
# # #                 else:
# # #                     # If it's a single tuple, clean it and wrap in list
# # #                     tile_cands[t_idx] = [clean_tuple(val)]

# # #             # 2. Update the BoardTracker (Temporal Memory)
# # #             is_stable, _, _ = board_mem.step(tile_cands, frame_id)

# # #             # 3. Overwrite tile_to_piece with the STABLE state
# # #             tile_to_piece = {}
# # #             for i in range(64):
# # #                 stable_p = board_mem.board.get_piece(i)
# # #                 if stable_p:
# # #                     tile_to_piece[i] = stable_p



            
# # #             # --- DISPLAY ---
# # #             if SAVE_DEBUG_FRAMES:
# # #                 vis_img = original_img.copy()
# # #                 draw_grid_lines(vis_img, grid_points_global)
# # #                 visualize_pieces(vis_img, tile_to_piece, tile_centers_global)
# # #                 cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), vis_img)
            
# # #             total_ms = (t_pieces - t_start) * 1000.0
# # #             stats_total.append(total_ms)
            
# # #             print(f"Frame {frame_id:<10} | {(t_seg-t_start)*1000:<8.1f} | {(t_grid-t_parallel)*1000:<8.1f} | {(t_pieces-t_parallel)*1000 - (t_grid-t_parallel)*1000:<8.1f} | {total_ms:<8.1f} | OK")
            
# # #             if tile_to_piece:
# # #                 print_ascii_board(tile_to_piece, frame_id)

# # #             if frame_id % 100 == 0 and len(stats_total) > 0:
# # #                 result_queue.put_nowait(f"Status: {len(stats_total)} OK, {stats_skipped} skipped, {sum(stats_total)/len(stats_total):.1f}ms avg")

# # #         except Exception as e:
# # #             print(f"[AI] Error: {e}", flush=True); traceback.print_exc()
# # #             tracker.reset(); time.sleep(0.1)