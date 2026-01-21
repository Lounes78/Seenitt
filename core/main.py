"""
Chessboard Detection Pipeline with Performance Optimizations:

1. Vision Encoder Caching: Run the expensive Vision Encoder once per frame,
   then reuse features for both board and piece segmentation (decoder only).
   
2. Parallel CPU/GPU Execution: Grid Solver (CPU) and Piece Decoder (GPU) 
   run simultaneously using ThreadPool, hiding latency of the faster operation.
"""

import os
import glob
import cv2
import numpy as np
import time
import sys
import argparse
from multiprocessing.pool import ThreadPool
from src.segmentation import ChessboardSegmenter
from src.grid_solver import GridSolver
from src.filter import QualityFilter
from src.ChessboardState import ChessboardState
from src.tracker import BoardTracker 
from src.StockfishModule import StockfishModule

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../PieceClassifier")) 
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from PieceClassifier import PieceClassifier

CLASS_NAMES = [
    'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
    'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'
]

CLASS_TO_FEN = {
    "black-king": "k", "black-queen": "q", "black-rook": "r", "black-bishop": "b", "black-knight": "n", "black-pawn": "p",
    "white-king": "K", "white-queen": "Q", "white-rook": "R", "white-bishop": "B", "white-knight": "N", "white-pawn": "P",
}


MAX_CHANGED_TILES = 4  

MIN_CONF = 0.95
MIN_MARGIN = 0.10

TOTAL_TIME_LIMIT_MS = 1000.0 

def safe_bbox_from_mask(mask_u8):
    # mask_u8: 2D uint8 or bool
    if mask_u8.dtype != np.uint8:
        mask_u8 = (mask_u8 > 0).astype(np.uint8)

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def draw_label(img_bgr, text, x1, y1, x2, y2, color=(255, 255, 255)):
    # place label above the bbox if possible
    y_text = max(0, y1 - 8)
    cv2.putText(img_bgr, text, (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2, cv2.LINE_AA)

VALID_PIECES = set("PNBRQKpnbrqk")

def render_virtual_board(board, square=70, margin=30,
                         show_conf=False, min_conf=0.0,
                         show_color_tag=False,
                         best_move=None):
    """
    board: ChessboardState
    returns: BGR image (uint8)
    Indexing assumed: tile_idx = r*8 + c (r=0 top row)
    """
    W = 8 * square + 2 * margin
    H = 8 * square + 2 * margin
    img = np.zeros((H, W, 3), dtype=np.uint8) + 245

    # draw squares
    for r in range(8):
        for c in range(8):
            x1 = margin + c * square
            y1 = margin + r * square
            x2 = x1 + square
            y2 = y1 + square
            is_dark = (r + c) % 2 == 1
            color = (180, 180, 180) if is_dark else (235, 235, 235)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    # grid lines
    for k in range(9):
        x = margin + k * square
        y = margin + k * square
        cv2.line(img, (x, margin), (x, margin + 8 * square), (120, 120, 120), 1)
        cv2.line(img, (margin, y), (margin + 8 * square, y), (120, 120, 120), 1)

    # pieces
    for idx in range(64):
        st = board.tiles[idx]
        p = st.piece

        if not p or p not in VALID_PIECES:
            continue
        if st.conf < max(min_conf, 0.0):
            continue

        r, c = divmod(idx, 8)
        cx = margin + c * square + square // 2
        cy = margin + r * square + square // 2

        is_white = p.isupper()

        # colores intuitivos
        text_color = (240, 240, 240) if is_white else (30, 30, 30)
        outline    = (30, 30, 30)     if is_white else (240, 240, 240)

        label = p
        if show_color_tag:
            label += "(W)" if is_white else "(B)"
        if show_conf:
            label += f"{st.conf:.2f}"

        cv2.putText(img, label, (cx - 18, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, outline, 4, cv2.LINE_AA)
        cv2.putText(img, label, (cx - 18, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)

    cv2.putText(img, "MEMORY BOARD", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    
    if best_move:
        cv2.putText(img, f"Best Move: {best_move}", (W - 220, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        #we  highlight the move on the board 
        from_square = best_move[0:2]
        to_square = best_move[2:4]
        file_to_col = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
        rank_to_row = {'1':7,'2':6,'3':5,'4':4,'5':3,'6':2,'7':1,'8':0}
        from_col = file_to_col[from_square[0]]
        from_row = rank_to_row[from_square[1]]
        to_col = file_to_col[to_square[0]]
        to_row = rank_to_row[to_square[1]]
        fx1 = margin + from_col * square
        fy1 = margin + from_row * square
        fx2 = fx1 + square
        fy2 = fy1 + square
        tx1 = margin + to_col * square
        ty1 = margin + to_row * square
        tx2 = tx1 + square
        ty2 = ty1 + square
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (0, 255, 0), 4)
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 0, 255), 4)




    return img


def hstack_images(left_bgr, right_bgr):
    # make same height
    h1, w1 = left_bgr.shape[:2]
    h2, w2 = right_bgr.shape[:2]
    if h1 != h2:
        scale = h1 / float(h2)
        new_w = int(w2 * scale)
        right_bgr = cv2.resize(right_bgr, (new_w, h1), interpolation=cv2.INTER_AREA)
    return np.hstack([left_bgr, right_bgr])


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
        piece_classifier = PieceClassifier(load_model_path = '../PieceClassifier/models/best_0.0555.pt',
                                           data_path =("../PieceClassifier/dataset/dataset1", "../PieceClassifier/dataset/dataset2"))
        sf_module = StockfishModule(path_stockfish='./src/stockfish-windows')
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    board = ChessboardState(inc=0.1,dec=0.05)
    board.init_standard()
    tracker = BoardTracker(
                            board=board,
                            max_changed_tiles=MAX_CHANGED_TILES,
                            diff_conf_thr=0.60,
                            min_keep=0.35,
                        )
    frame_idx = 0

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
    
    # Temporal tracking state
    prev_gray = None
    prev_grid_points = None
    prev_crop_bbox = None  # (x, y, w, h) of previous crop
    tracking_enabled = False
    
    print(f"{'FILENAME':<25} | {'SEG':<8} | {'GRID':<8} | {'PIECES':<8} | {'TOTAL':<8} | {'STATUS'}")
    print("-" * 95)

    for frame_idx,img_path in enumerate(image_files):
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
            grid_solved = False
            tracking_used = False
            curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            # Check if we can use optical flow tracking
            if (tracking_enabled and prev_gray is not None and prev_grid_points is not None 
                and prev_crop_bbox is not None):
                
                # Check if crop position is stable (board hasn't moved much)
                prev_x, prev_y, prev_w, prev_h = prev_crop_bbox
                dx = abs(x - prev_x)
                dy = abs(y - prev_y)
                dw = abs(w - prev_w)
                dh = abs(h - prev_h)
                
                # Check if crop size is the same (required for optical flow)
                size_match = (curr_gray.shape == prev_gray.shape)
                
                # If crop changed significantly or size mismatch, disable tracking
                if dx < 50 and dy < 50 and dw < 50 and dh < 50 and size_match:
                    try:
                        # FAST: Track grid points using Optical Flow (~2ms)
                        tracked_points, status, err = cv2.calcOpticalFlowPyrLK(
                            prev_gray, curr_gray, prev_grid_points, None,
                            winSize=(21, 21), maxLevel=3,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                        )
                        
                        # Check tracking quality: ALL 81 points must be tracked (9x9 grid)
                        num_tracked = np.sum(status)
                        
                        # Strict requirement: need all 81 points for reshaping
                        if num_tracked == 81 and tracked_points.shape[0] == 81:
                            # Successfully tracked all points!
                            grid_points_local = tracked_points.reshape(81, 2)
                            
                            # Reconstruct tile centers from grid (8x8 grid = 64 tile centers)
                            grid_9x9 = grid_points_local.reshape(9, 9, 2)
                            tile_centers_local = []
                            for row in range(8):
                                for col in range(8):
                                    tl = grid_9x9[row, col]
                                    tr = grid_9x9[row, col+1]
                                    bl = grid_9x9[row+1, col]
                                    br = grid_9x9[row+1, col+1]
                                    center = (tl + tr + bl + br) / 4.0
                                    tile_centers_local.append(center)
                            tile_centers_local = np.array(tile_centers_local)
                            
                            grid_solved = True
                            tracking_used = True
                            success = True
                            stats = {'method': 'optical_flow_tracking'}
                        else:
                            # Lost some points, fall back to solver
                            tracking_enabled = False
                    except Exception as track_err:
                        # Optical flow failed, fall back to solver
                        tracking_enabled = False
                else:
                    # Crop shifted too much or size mismatch, disable tracking
                    tracking_enabled = False

            # --- PARALLEL EXECUTION: Grid Solver (CPU) || Piece Decoder (GPU) ---
            
            # Define wrapper for piece segmentation to run in background thread
            def run_piece_decoder():
                return segmenter.decode_from_features(vision_features, original_img, "chess pieces")
            
            # Launch Piece Decoder on GPU in background thread
            pool = ThreadPool(processes=1)
            async_pieces = pool.apply_async(run_piece_decoder)
            
            # Run Grid Solver on CPU in main thread (parallel with GPU decoder)
            # SKIP if we already tracked the grid
            t_parallel_start = time.perf_counter()
            
            if not grid_solved:
                # SLOW: Full solver (~150ms)
                remaining_ms = TOTAL_TIME_LIMIT_MS - seg_ms
                warped_grid, grid_points_local, tile_centers_local, success, stats = solver.solve(
                    crop_img, crop_mask, time_limit_ms=remaining_ms
                )
                
                if success:
                    # Enable tracking for next frame
                    tracking_enabled = True
                    prev_grid_points = grid_points_local.reshape(-1, 1, 2).astype(np.float32)
                    prev_gray = curr_gray
                    prev_crop_bbox = curr_crop_bbox
            
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
                tracking_enabled = False
                prev_gray = None
                prev_grid_points = None
                prev_crop_bbox = None
                continue
            
            # Wait for Piece Decoder to finish and retrieve results
            piece_masks_list, _, _ = async_pieces.get()
            pool.close()
            pool.join()
            
            t_pieces_end = time.perf_counter()
            pieces_ms = max((t_pieces_end - t_parallel_start) * 1000.0 - grid_ms, 0.0)
            total_ms = (t_pieces_end - t_start) * 1000.0
            
            # --- END PARALLEL EXECUTION ---

            # 7. Visualization & Mapping
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

            # B. Process Pieces - ROBUST "Base-Centroid" Method
            # Note: piece_masks_list are full-size masks (same dimensions as original_img)
            # We crop them to the board region for piece detection
            
            # Define board boundary from grid corners (top-left, top-right, bottom-right, bottom-left)
            grid_corners = np.array([
                grid_points_global[0],      # top-left (0,0)
                grid_points_global[8],      # top-right (0,8)
                grid_points_global[80],     # bottom-right (8,8)
                grid_points_global[72]      # bottom-left (8,0)
            ], dtype=np.float32)
            
            piece_data = []
            
            for i, full_piece_mask in enumerate(piece_masks_list):
                # Crop the piece mask to board region
                local_piece_mask = full_piece_mask[y:y+h, x:x+w]
                
                # Get bounding box of the piece within the board crop
                x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
                if w_p == 0 or h_p == 0: continue

                # --- ROBUST BASE DETECTION START ---
                # 1. Isolate the bottom 20% of the piece (the "base")
                # This focuses strictly on where the piece touches the board
                base_ratio = 0.20
                slice_h = int(h_p * base_ratio)
                
                # Safety check for very small pieces
                if slice_h < 1: slice_h = 1
                
                y_slice_start = y_p + h_p - slice_h
                y_slice_end = y_p + h_p
                
                # Extract the base mask
                base_mask = local_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
                
                # 2. Calculate Centroid of the BASE only
                M = cv2.moments(base_mask)
                
                if M["m00"] > 0:
                    cX_base = int(M["m10"] / M["m00"])
                    cY_base = int(M["m01"] / M["m00"])
                    
                    # Map back to Global Coordinates
                    # Global X = Board_Crop_X + Piece_BBox_X + Base_Centroid_X
                    # Global Y = Board_Crop_Y + Slice_Start_Y + Base_Centroid_Y
                    pc_global = np.array([x + x_p + cX_base, y + y_slice_start + cY_base])
                    area = cv2.moments(local_piece_mask)["m00"]  # Use full area for deduplication
                else:
                    # Fallback: Bottom-center of bounding box
                    pc_global = np.array([x + x_p + w_p // 2, y + y_p + h_p])
                    area = 0
                # --- ROBUST BASE DETECTION END ---

                # FILTER: Check if contact point is inside board boundary
                is_inside = cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False)
                if is_inside < 0:  # Point is outside the polygon
                    continue

                # Find closest tile center
                dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
                closest_idx = np.argmin(dists)
                closest_dist = dists[closest_idx]
                
                piece_data.append({
                    'idx': i,
                    'mask': full_piece_mask,
                    'centroid': pc_global,
                    'tile_idx': closest_idx,
                    'distance': closest_dist,
                    'area': area
                })
            
            # Deduplicate: keep only one piece per tile (largest area)
            tile_to_piece = {}
            for piece in piece_data:
                tile_idx = piece['tile_idx']
                if tile_idx not in tile_to_piece or piece['area'] > tile_to_piece[tile_idx]['area']:
                    tile_to_piece[tile_idx] = piece
            
            # C. Visualize deduplicated pieces and build obs
            #obs = {i: (None, 1.0) for i in range(64)}
            obs = {}
            tile_cands = {}
            for piece in tile_to_piece.values():
                i = piece['idx']
                full_piece_mask = piece['mask']
                pc_global = piece['centroid']
                tile_idx = int(piece['tile_idx'])
                
                closest_center = tile_centers_global[tile_idx]

                bbox = safe_bbox_from_mask(full_piece_mask)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox

                pad = 6
                H_img, W_img = original_img.shape[:2]
                x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
                x2p = min(W_img - 1, x2 + pad); y2p = min(H_img - 1, y2 + pad)

                crop_bgr = original_img[y1p:y2p, x1p:x2p].copy()

                #pred_idx, pred_conf = piece_classifier.predict(crop_bgr)
                pred_idxs, pred_confs = piece_classifier.predict_topk(crop_bgr,12)
                
                cands = []
                for ci, cp in zip(pred_idxs, pred_confs):
                    fen = CLASS_TO_FEN[CLASS_NAMES[ci]]
                    cands.append((fen, float(cp)))

                
                if len(cands) >= 2:
                    if cands[0][1] < MIN_CONF or (cands[0][1] - cands[1][1]) < MIN_MARGIN:
                        continue  
                else:
                    if cands[0][1] < MIN_CONF:
                        continue

                tile_cands[tile_idx] = cands


                # Unique Color
                hue = int((i * 137.508) % 180)
                hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                piece_color = tuple(map(int, rgb_color))

                # Overlay using full-size mask
                colored_layer = np.zeros_like(vis_img)
                colored_layer[:] = piece_color

                mask_indices = full_piece_mask > 0
                if np.any(mask_indices):
                    vis_img[mask_indices] = cv2.addWeighted(
                        vis_img[mask_indices], 0.6,
                        colored_layer[mask_indices], 0.4,
                        0
                    )

                # Draw Vector
                pt1 = tuple(pc_global.astype(int))
                pt2 = tuple(closest_center.astype(int))

                cv2.line(vis_img, pt1, pt2, piece_color, 2)
                cv2.circle(vis_img, pt1, 6, (255, 255, 255), -1)
                cv2.circle(vis_img, pt1, 4, piece_color, -1)

            # Update persistent board state (only on OK frames)
            #board.update(obs, frame_idx)
            #board.set_from_observation(obs, frame_idx)
            accept_update, obs, diffs = tracker.step(tile_cands, frame_idx)
            sf_module.set_position_from_state(tracker.board)
            best_move = sf_module.get_best_move(frame_idx)


            if not accept_update:
                print(f"[REJECT] frame {frame_idx} too many diffs: {len(diffs)} tiles")

            # Optional: draw the persistent board state on top
            #board.draw_overlay_visible(vis_img, tile_centers_global, obs)
            memory_img = render_virtual_board(tracker.board, square=70, show_conf=False, min_conf=0.0,best_move = best_move)

            combo = hstack_images(vis_img, memory_img)


            cv2.imwrite(os.path.join(output_dir, filename), combo)
            
            stats_total.append(total_ms)
            status_suffix = " [TRACKED]" if tracking_used else ""
            print(f"{filename:<25} | {seg_ms:<8.1f} | {grid_ms:<8.1f} | {pieces_ms:<8.1f} | {total_ms:<8.1f} | OK{status_suffix}")

        except Exception as e:
            msg = f"ERROR: {str(e)}"
            print(f"{filename:<25} | {msg}")
            log_file.write(f"{filename}: {msg}\n")
            stats_skipped += 1
            
            # Reset tracking on error
            tracking_enabled = False
            prev_gray = None
            prev_grid_points = None
            prev_crop_bbox = None
        
        #frame_idx+=1

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
