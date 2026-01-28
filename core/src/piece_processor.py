"""
Piece detection and assignment logic.
Handles base-centroid detection and tile assignment with deduplication.
"""

import cv2
import numpy as np

def process_pieces(piece_masks_list, crop_bbox, grid_points_global, tile_centers_global, move_count=0):
    """
    Process chess piece masks and assign them to board tiles.
    """
    x, y, w, h = crop_bbox
    
    # Define board boundary from grid corners
    grid_corners = np.array([
        grid_points_global[0],      # top-left
        grid_points_global[8],      # top-right
        grid_points_global[80],     # bottom-right
        grid_points_global[72]      # bottom-left
    ], dtype=np.float32)
    
    piece_data = []

    for i, full_piece_mask in enumerate(piece_masks_list):
        # 1. Restore Original Cropping Logic
        # Crop the piece mask to the board region (or the expanded region passed in crop_bbox)
        local_piece_mask = full_piece_mask[y:y+h, x:x+w]
        
        # Get bounding box relative to the crop
        x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
        
        # Basic validation
        if w_p == 0 or h_p == 0:
            continue

        # Calculate Area
        area = cv2.moments(local_piece_mask)["m00"]
        if area <= 0: 
            continue

        # --- BASE DETECTION (Bottom 20%) ---
        base_ratio = 0.20
        slice_h = max(1, int(h_p * base_ratio))
        y_slice_start = y_p + h_p - slice_h
        y_slice_end = y_p + h_p
        
        base_mask = local_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
        M = cv2.moments(base_mask)
        
        # 2. Restore Original Centroid Math (Local + Offset)
        if M["m00"] > 0:
            cX_base = int(M["m10"] / M["m00"])
            cY_base = int(M["m01"] / M["m00"])
            # The original formula you wanted:
            pc_global = np.array([x + x_p + cX_base, y + y_slice_start + cY_base])
        else:
            # Fallback: Bottom-center of bounding box
            pc_global = np.array([x + x_p + w_p // 2, y + y_p + h_p])

        # 3. DISABLE "Outside Grid" Deletion
        # The code below is commented out so we keep pieces outside the grid lines
        # if cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False) < 0:
        #    continue

        # Find closest tile center
        dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
        closest_idx = np.argmin(dists)
        
        piece_data.append({
            'idx': i,
            'mask': full_piece_mask,
            'centroid': pc_global,
            'tile_idx': closest_idx,
            'area': area
        })
    
    # Deduplicate: keep only one piece per tile (largest area)
    tile_to_piece = {}
    for piece in piece_data:
        tile_idx = piece['tile_idx']
        if tile_idx not in tile_to_piece or piece['area'] > tile_to_piece[tile_idx]['area']:
            tile_to_piece[tile_idx] = piece
    
    return tile_to_piece



# """
# Piece detection and assignment logic.
# Handles base-centroid detection and tile assignment with deduplication.
# """

# import cv2
# import numpy as np

# def process_pieces(piece_masks_list, crop_bbox, grid_points_global, tile_centers_global, move_count=0):
#     """
#     Process chess piece masks and assign them to board tiles.
#     NOW MODIFIED: Includes pieces outside the grid boundary (marked as outliers).
    
#     Args:
#         piece_masks_list: List of binary masks for each detected piece
#         crop_bbox: Tuple (x, y, w, h) of the board crop region
#         grid_points_global: Array of 81 grid intersection points (9x9)
#         tile_centers_global: Array of 64 tile center points (8x8)
#         move_count: The current move number (kept for compatibility, logic removed)
    
#     Returns:
#         dict: Mapping from tile_idx to piece data dictionary
#     """
#     x, y, w, h = crop_bbox
    
#     # Define board boundary from grid corners
#     grid_corners = np.array([
#         grid_points_global[0],      # top-left
#         grid_points_global[8],      # top-right
#         grid_points_global[80],     # bottom-right
#         grid_points_global[72]      # bottom-left
#     ], dtype=np.float32)
    
#     piece_data = []

#     for i, full_piece_mask in enumerate(piece_masks_list):
#         # We calculate moments on the FULL mask to support pieces outside the crop
#         M = cv2.moments(full_piece_mask)
        
#         if M["m00"] <= 0: continue
        
#         # Get bounding box (Global)
#         x_p, y_p, w_p, h_p = cv2.boundingRect(full_piece_mask)
#         if w_p == 0 or h_p == 0: continue

#         # --- BASE DETECTION (Bottom 20%) ---
#         # We want the base of the piece, not the visual center
#         base_ratio = 0.20
#         slice_h = max(1, int(h_p * base_ratio))
#         y_slice_start = y_p + h_p - slice_h
#         y_slice_end = y_p + h_p
        
#         # Slice the global mask to get the base area
#         base_mask = full_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
#         M_base = cv2.moments(base_mask)
        
#         if M_base["m00"] > 0:
#             cX_base = int(M_base["m10"] / M_base["m00"]) + x_p
#             cY_base = int(M_base["m01"] / M_base["m00"]) + y_slice_start
#             pc_global = np.array([cX_base, cY_base])
#         else:
#             # Fallback: Bottom-center of bounding box
#             pc_global = np.array([x_p + w_p // 2, y_p + h_p])

#         # --- MODIFIED: REMOVED BOUNDARY CHECK ---
#         # Previously, we deleted pieces outside 'grid_corners'.
#         # Now we calculate if it's an outlier but KEEP it.
#         is_outside = cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False) < 0

#         # Find closest tile center (even if far away)
#         dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
#         closest_idx = np.argmin(dists)
        
#         piece_data.append({
#             'idx': i,
#             'mask': full_piece_mask,
#             'centroid': pc_global,
#             'tile_idx': closest_idx,
#             'area': M["m00"],
#             'is_outlier': is_outside # Flag for visualization/logic
#         })
    
#     # Deduplicate: keep only one piece per tile (largest area)
#     tile_to_piece = {}
#     for piece in piece_data:
#         tile_idx = piece['tile_idx']
        
#         # Collision handling:
#         if tile_idx not in tile_to_piece:
#              tile_to_piece[tile_idx] = piece
#         else:
#             # If we have a collision, prefer the one with larger area
#             if piece['area'] > tile_to_piece[tile_idx]['area']:
#                 tile_to_piece[tile_idx] = piece
    
#     return tile_to_piece





# # """
# # Piece detection and assignment logic.
# # Handles base-centroid detection and tile assignment with deduplication.
# # """

# # import cv2
# # import numpy as np

# # def process_pieces(piece_masks_list, crop_bbox, grid_points_global, tile_centers_global, move_count=0):
# #     """
# #     Process chess piece masks and assign them to board tiles.
    
# #     Args:
# #         piece_masks_list: List of binary masks for each detected piece
# #         crop_bbox: Tuple (x, y, w, h) of the board crop region
# #         grid_points_global: Array of 81 grid intersection points (9x9)
# #         tile_centers_global: Array of 64 tile center points (8x8)
# #         move_count: The current move number (kept for compatibility, logic removed)
    
# #     Returns:
# #         dict: Mapping from tile_idx to piece data dictionary
# #     """
# #     x, y, w, h = crop_bbox
    
# #     # Define board boundary from grid corners to ignore noise outside board
# #     grid_corners = np.array([
# #         grid_points_global[0],      # top-left
# #         grid_points_global[8],      # top-right
# #         grid_points_global[80],     # bottom-right
# #         grid_points_global[72]      # bottom-left
# #     ], dtype=np.float32)
    
# #     piece_data = []

# #     for i, full_piece_mask in enumerate(piece_masks_list):
# #         # Crop the piece mask to board region
# #         local_piece_mask = full_piece_mask[y:y+h, x:x+w]
        
# #         # Get bounding box
# #         x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
        
# #         # Basic validation: ignore empty or zero-dimension masks
# #         if w_p == 0 or h_p == 0:
# #             continue

# #         # Calculate Area (for deduplication later)
# #         area = cv2.moments(local_piece_mask)["m00"]
# #         if area <= 0: 
# #             continue

# #         # --- BASE DETECTION (Bottom 20%) ---
# #         base_ratio = 0.20
# #         slice_h = max(1, int(h_p * base_ratio))
# #         y_slice_start = y_p + h_p - slice_h
# #         y_slice_end = y_p + h_p
        
# #         base_mask = local_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
# #         M = cv2.moments(base_mask)
        
# #         if M["m00"] > 0:
# #             cX_base = int(M["m10"] / M["m00"])
# #             cY_base = int(M["m01"] / M["m00"])
# #             pc_global = np.array([x + x_p + cX_base, y + y_slice_start + cY_base])
# #         else:
# #             # Fallback: Bottom-center of bounding box
# #             pc_global = np.array([x + x_p + w_p // 2, y + y_p + h_p])

# #         # Check if point is inside board boundary
# #         if cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False) < 0:
# #             continue

# #         # Find closest tile center
# #         dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
# #         closest_idx = np.argmin(dists)
        
# #         # [REMOVED] Row 3 exclusion logic deleted here

# #         piece_data.append({
# #             'idx': i,
# #             'mask': full_piece_mask,
# #             'centroid': pc_global,
# #             'tile_idx': closest_idx,
# #             'area': area
# #         })
    
# #     # Deduplicate: keep only one piece per tile (largest area)
# #     tile_to_piece = {}
# #     for piece in piece_data:
# #         tile_idx = piece['tile_idx']
# #         if tile_idx not in tile_to_piece or piece['area'] > tile_to_piece[tile_idx]['area']:
# #             tile_to_piece[tile_idx] = piece
    
# #     return tile_to_piece



# # # """
# # # Piece detection and assignment logic.
# # # Handles base-centroid detection and tile assignment with deduplication.
# # # """

# # # import cv2
# # # import numpy as np

# # # def process_pieces(piece_masks_list, crop_bbox, grid_points_global, tile_centers_global, move_count=0):
# # #     """
# # #     Process chess piece masks and assign them to board tiles.
# # #     Includes specific opening-game discard rule for Row 3.
    
# # #     Args:
# # #         piece_masks_list: List of binary masks for each detected piece
# # #         crop_bbox: Tuple (x, y, w, h) of the board crop region
# # #         grid_points_global: Array of 81 grid intersection points (9x9)
# # #         tile_centers_global: Array of 64 tile center points (8x8)
# # #         move_count: The current move number of the game (default 0)
    
# # #     Returns:
# # #         dict: Mapping from tile_idx to piece data dictionary
# # #     """
# # #     x, y, w, h = crop_bbox
    
# # #     # Define board boundary from grid corners to ignore noise outside board
# # #     grid_corners = np.array([
# # #         grid_points_global[0],      # top-left
# # #         grid_points_global[8],      # top-right
# # #         grid_points_global[80],     # bottom-right
# # #         grid_points_global[72]      # bottom-left
# # #     ], dtype=np.float32)
    
# # #     piece_data = []

# # #     for i, full_piece_mask in enumerate(piece_masks_list):
# # #         # Crop the piece mask to board region
# # #         local_piece_mask = full_piece_mask[y:y+h, x:x+w]
        
# # #         # Get bounding box
# # #         x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
        
# # #         # Basic validation: ignore empty or zero-dimension masks
# # #         if w_p == 0 or h_p == 0:
# # #             continue

# # #         # Calculate Area (for deduplication later)
# # #         area = cv2.moments(local_piece_mask)["m00"]
# # #         if area <= 0: 
# # #             continue

# # #         # --- BASE DETECTION (Bottom 20%) ---
# # #         base_ratio = 0.20
# # #         slice_h = max(1, int(h_p * base_ratio))
# # #         y_slice_start = y_p + h_p - slice_h
# # #         y_slice_end = y_p + h_p
        
# # #         base_mask = local_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
# # #         M = cv2.moments(base_mask)
        
# # #         if M["m00"] > 0:
# # #             cX_base = int(M["m10"] / M["m00"])
# # #             cY_base = int(M["m01"] / M["m00"])
# # #             pc_global = np.array([x + x_p + cX_base, y + y_slice_start + cY_base])
# # #         else:
# # #             # Fallback: Bottom-center of bounding box
# # #             pc_global = np.array([x + x_p + w_p // 2, y + y_p + h_p])

# # #         # Check if point is inside board boundary
# # #         if cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False) < 0:
# # #             continue

# # #         # Find closest tile center
# # #         dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
# # #         closest_idx = np.argmin(dists)
        
# # #         # --- NEW LOGIC: Row 3 Exclusion ---
# # #         # Calculate row index (0-7).
# # #         row, col = divmod(closest_idx, 8)
        
# # #         # OPENING FILTER: 
# # #         # If we are in the first 10 moves, discard detections strictly on row index 3.
# # #         # This prevents ghost detections often caused by shadows/angles in this specific row.
# # #         if move_count <= 10 and row == 3:
# # #             # Optionally log: print(f"Discarding detection at row {row} col {col}")
# # #             continue

# # #         piece_data.append({
# # #             'idx': i,
# # #             'mask': full_piece_mask,
# # #             'centroid': pc_global,
# # #             'tile_idx': closest_idx,
# # #             'area': area
# # #         })
    
# # #     # Deduplicate: keep only one piece per tile (largest area)
# # #     tile_to_piece = {}
# # #     for piece in piece_data:
# # #         tile_idx = piece['tile_idx']
# # #         if tile_idx not in tile_to_piece or piece['area'] > tile_to_piece[tile_idx]['area']:
# # #             tile_to_piece[tile_idx] = piece
    
# # #     return tile_to_piece






# # # """
# # # Piece detection and assignment logic.
# # # Handles base-centroid detection and tile assignment with deduplication.
# # # """

# # # import cv2
# # # import numpy as np


# # # def process_pieces(piece_masks_list, crop_bbox, grid_points_global, tile_centers_global, min_mask_ratio=0.25, min_aspect_ratio=0.85, max_aspect_ratio=3.5):
# # #     """
# # #     Process chess piece masks and assign them to board tiles.
    
# # #     Args:
# # #         piece_masks_list: List of binary masks (full image size) for each detected piece
# # #         crop_bbox: Tuple (x, y, w, h) of the board crop region
# # #         grid_points_global: Array of 81 grid intersection points (9x9)
# # #         tile_centers_global: Array of 64 tile center points (8x8)
# # #         min_mask_ratio: Minimum ratio of mask area to median mask area (default 0.25 = 25%, aggressive)
# # #         min_aspect_ratio: Hard minimum height/width ratio (default 0.85 = catch blobby shapes)
# # #         max_aspect_ratio: Hard maximum height/width ratio (default 3.5 = reject extreme outliers)
    
# # #     Returns:
# # #         dict: Mapping from tile_idx to piece data dictionary
# # #     """
# # #     x, y, w, h = crop_bbox
    
# # #     # Define board boundary from grid corners
# # #     grid_corners = np.array([
# # #         grid_points_global[0],      # top-left (0,0)
# # #         grid_points_global[8],      # top-right (0,8)
# # #         grid_points_global[80],     # bottom-right (8,8)
# # #         grid_points_global[72]      # bottom-left (8,0)
# # #     ], dtype=np.float32)
    
# # #     # --- STEP 1: Calculate mask metrics for statistical filtering ---
# # #     mask_data = []
    
# # #     for i, full_piece_mask in enumerate(piece_masks_list):
# # #         local_piece_mask = full_piece_mask[y:y+h, x:x+w]
# # #         area = cv2.moments(local_piece_mask)["m00"]
        
# # #         if area > 0:
# # #             x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
# # #             if w_p > 0 and h_p > 0:
# # #                 aspect_ratio = h_p / w_p
# # #                 mask_data.append({
# # #                     'idx': i,
# # #                     'area': area,
# # #                     'aspect_ratio': aspect_ratio,
# # #                     'width': w_p,
# # #                     'height': h_p
# # #                 })
    
# # #     # Need at least a few pieces to calculate statistics
# # #     if len(mask_data) < 3:
# # #         return {}
    
# # #     # Calculate statistical thresholds
# # #     areas = [m['area'] for m in mask_data]
# # #     aspect_ratios = [m['aspect_ratio'] for m in mask_data]
    
# # #     median_area = np.median(areas)
# # #     median_aspect = np.median(aspect_ratios)
    
# # #     # For aspect ratio, calculate standard deviation to identify outliers
# # #     std_aspect = np.std(aspect_ratios)
    
# # #     # Thresholds
# # #     min_area_threshold = median_area * min_mask_ratio
    
# # #     # Simple hard thresholds for aspect ratio - no statistical filtering
# # #     # This is more predictable and avoids over-filtering
# # #     print(f"[PieceProcessor] Area - Median: {median_area:.0f}, Min threshold: {min_area_threshold:.0f}")
# # #     print(f"[PieceProcessor] Aspect - Median: {median_aspect:.2f}, StdDev: {std_aspect:.2f}")
# # #     print(f"[PieceProcessor] Aspect - Valid range: [{min_aspect_ratio:.2f}, {max_aspect_ratio:.2f}]")
    
# # #     # Create lookup for quick filtering
# # #     mask_metrics = {m['idx']: m for m in mask_data}
    
# # #     # --- STEP 2: Process pieces with intelligent occlusion-aware filtering ---
# # #     piece_data = []
# # #     filtered_size_count = 0
# # #     filtered_occluded_count = 0
    
# # #     # First pass: collect all valid pieces (those that pass basic checks)
# # #     preliminary_pieces = []
    
# # #     for i, full_piece_mask in enumerate(piece_masks_list):
# # #         # Skip if we didn't get metrics for this mask
# # #         if i not in mask_metrics:
# # #             continue
        
# # #         metrics = mask_metrics[i]
        
# # #         # FILTER 1: Reject abnormally small masks (likely occluded/partial detections)
# # #         if metrics['area'] < min_area_threshold:
# # #             filtered_size_count += 1
# # #             print(f"[PieceProcessor] ⚠️  FILTERED mask {i}: area={metrics['area']:.0f} < {min_area_threshold:.0f} (too small)")
# # #             continue
        
# # #         # Crop the piece mask to board region
# # #         local_piece_mask = full_piece_mask[y:y+h, x:x+w]
        
# # #         # Get bounding box of the piece within the board crop
# # #         x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
# # #         if w_p == 0 or h_p == 0:
# # #             continue

# # #         # --- ROBUST BASE DETECTION ---
# # #         # Isolate the bottom 20% of the piece (the "base")
# # #         base_ratio = 0.20
# # #         slice_h = max(1, int(h_p * base_ratio))  # Safety check for small pieces
        
# # #         y_slice_start = y_p + h_p - slice_h
# # #         y_slice_end = y_p + h_p
        
# # #         # Extract the base mask
# # #         base_mask = local_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
        
# # #         # Calculate Centroid of the BASE only
# # #         M = cv2.moments(base_mask)
        
# # #         if M["m00"] > 0:
# # #             cX_base = int(M["m10"] / M["m00"])
# # #             cY_base = int(M["m01"] / M["m00"])
            
# # #             # Map to global coordinates
# # #             pc_global = np.array([x + x_p + cX_base, y + y_slice_start + cY_base])
# # #             area = metrics['area']  # Use pre-calculated area
# # #         else:
# # #             # Fallback: Bottom-center of bounding box
# # #             pc_global = np.array([x + x_p + w_p // 2, y + y_p + h_p])
# # #             area = metrics['area']

# # #         # FILTER: Check if contact point is inside board boundary
# # #         is_inside = cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False)
# # #         if is_inside < 0:  # Point is outside
# # #             continue

# # #         # Find closest tile center
# # #         dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
# # #         closest_idx = np.argmin(dists)
# # #         closest_dist = dists[closest_idx]
        
# # #         preliminary_pieces.append({
# # #             'mask_idx': i,
# # #             'idx': i,
# # #             'mask': full_piece_mask,
# # #             'centroid': pc_global,
# # #             'tile_idx': closest_idx,
# # #             'distance': closest_dist,
# # #             'area': area,
# # #             'aspect_ratio': metrics['aspect_ratio']
# # #         })
    
# # #     # --- INTELLIGENT OCCLUSION FILTERING ---
# # #     # Check both size and aspect ratio candidates - filter if there's a piece blocking the view
# # #     # ONLY for bottom half (white's side) - check the 2 rows closer to row 1
# # #     for piece in preliminary_pieces:
# # #         aspect = piece['aspect_ratio']
# # #         area = piece['area']
# # #         tile_idx = piece['tile_idx']
        
# # #         # Check if this piece is suspicious (small area OR unusual aspect ratio)
# # #         is_suspicious_size = area < min_area_threshold * 1.5  # Check pieces up to 1.5x threshold
# # #         is_suspicious_aspect = aspect < min_aspect_ratio or aspect > max_aspect_ratio
# # #         is_suspicious = is_suspicious_size or is_suspicious_aspect
        
# # #         if is_suspicious:
# # #             # Calculate the tiles in front - ONLY for bottom half
# # #             row, col = divmod(tile_idx, 8)
            
# # #             # Display row = 8 - internal_row (row 0 = display 8, row 7 = display 1)
# # #             display_row = 8 - row
            
# # #             # ONLY check bottom half (display rows 1-4, internal rows 4-7)
# # #             if row >= 4:  # Bottom half (white's perspective)
# # #                 # Check 2 rows with LOWER display numbers (higher internal row numbers)
# # #                 # For b3 (display=3, internal=5): check b2 (internal=6) and b1 (internal=7)
# # #                 blocking_rows = [row + 1, row + 2]
                
# # #                 # Filter out invalid rows (off the board)
# # #                 blocking_rows = [r for r in blocking_rows if 0 <= r < 8]
                
# # #                 # Check if there's ANY piece in the 2 rows in front on the same column
# # #                 has_blocker = False
# # #                 blocker_tiles = []
                
# # #                 for blocking_row in blocking_rows:
# # #                     blocking_tile_idx = blocking_row * 8 + col
# # #                     if any(p['tile_idx'] == blocking_tile_idx for p in preliminary_pieces):
# # #                         has_blocker = True
# # #                         blocker_tiles.append(blocking_tile_idx)
                
# # #                 if has_blocker:
# # #                     # There's a piece in front - this is likely an occluded partial detection
# # #                     filtered_occluded_count += 1
# # #                     reason = f"small area={area:.0f}" if is_suspicious_size else f"aspect={aspect:.2f}"
# # #                     blocker_positions = ', '.join([f"{chr(97+b%8)}{8-b//8}" for b in blocker_tiles])
# # #                     print(f"[PieceProcessor] ⚠️  FILTERED mask {piece['mask_idx']}: {reason}, tile={chr(97+col)}{display_row} - occluded by piece(s) at {blocker_positions}")
# # #                     continue  # Skip this piece
# # #                 else:
# # #                     # No blocker - might be a legitimate piece with unusual characteristics
# # #                     reason = f"small area={area:.0f}" if is_suspicious_size else f"aspect={aspect:.2f}"
# # #                     print(f"[PieceProcessor] ℹ️  KEPT mask {piece['mask_idx']}: {reason}, tile={chr(97+col)}{display_row} (suspicious but no blocker)")
# # #             else:
# # #                 # Top half - no occlusion checking, just warn if suspicious
# # #                 if is_suspicious_size or is_suspicious_aspect:
# # #                     reason = f"small area={area:.0f}" if is_suspicious_size else f"aspect={aspect:.2f}"
# # #                     print(f"[PieceProcessor] ℹ️  KEPT mask {piece['mask_idx']}: {reason}, tile={chr(97+col)}{display_row} (top half, no occlusion check)")
        
# # #         # Piece passed all filters
# # #         piece_data.append(piece)
    
# # #     if filtered_size_count > 0:
# # #         print(f"[PieceProcessor] Filtered {filtered_size_count} small masks out of {len(piece_masks_list)}")
# # #     if filtered_occluded_count > 0:
# # #         print(f"[PieceProcessor] Filtered {filtered_occluded_count} occluded partial detections out of {len(piece_masks_list)}")
    
# # #     # Deduplicate: keep only one piece per tile (largest area)
# # #     tile_to_piece = {}
# # #     for piece in piece_data:
# # #         tile_idx = piece['tile_idx']
# # #         if tile_idx not in tile_to_piece or piece['area'] > tile_to_piece[tile_idx]['area']:
# # #             tile_to_piece[tile_idx] = piece
    
# # #     return tile_to_piece



# # # """
# # # Piece detection and assignment logic.
# # # Handles base-centroid detection and tile assignment with deduplication.
# # # """

# # # import cv2
# # # import numpy as np


# # # def process_pieces(piece_masks_list, crop_bbox, grid_points_global, tile_centers_global, min_mask_ratio=0.3):
# # #     """
# # #     Process chess piece masks and assign them to board tiles.
    
# # #     Args:
# # #         piece_masks_list: List of binary masks (full image size) for each detected piece
# # #         crop_bbox: Tuple (x, y, w, h) of the board crop region
# # #         grid_points_global: Array of 81 grid intersection points (9x9)
# # #         tile_centers_global: Array of 64 tile center points (8x8)
# # #         min_mask_ratio: Minimum ratio of mask area to median mask area (default 0.15 = 15%)
    
# # #     Returns:
# # #         dict: Mapping from tile_idx to piece data dictionary
# # #     """
# # #     x, y, w, h = crop_bbox
    
# # #     # Define board boundary from grid corners
# # #     grid_corners = np.array([
# # #         grid_points_global[0],      # top-left (0,0)
# # #         grid_points_global[8],      # top-right (0,8)
# # #         grid_points_global[80],     # bottom-right (8,8)
# # #         grid_points_global[72]      # bottom-left (8,0)
# # #     ], dtype=np.float32)
    
# # #     # --- STEP 1: Calculate mask areas for filtering ---
# # #     mask_areas = []
# # #     for full_piece_mask in piece_masks_list:
# # #         local_piece_mask = full_piece_mask[y:y+h, x:x+w]
# # #         area = cv2.moments(local_piece_mask)["m00"]
# # #         if area > 0:
# # #             mask_areas.append(area)
    
# # #     # Calculate median area and threshold
# # #     if len(mask_areas) == 0:
# # #         return {}
    
# # #     median_area = np.median(mask_areas)
# # #     min_area_threshold = median_area * min_mask_ratio
    
# # #     print(f"[PieceProcessor] Median mask area: {median_area:.0f}, Min threshold: {min_area_threshold:.0f}")
    
# # #     # --- STEP 2: Process pieces with size filtering ---
# # #     piece_data = []
# # #     filtered_count = 0
    
# # #     for i, full_piece_mask in enumerate(piece_masks_list):
# # #         # Crop the piece mask to board region
# # #         local_piece_mask = full_piece_mask[y:y+h, x:x+w]
        
# # #         # Get full area for filtering
# # #         full_area = cv2.moments(local_piece_mask)["m00"]
        
# # #         # FILTER: Reject abnormally small masks (likely occluded/partial detections)
# # #         if full_area < min_area_threshold:
# # #             filtered_count += 1
# # #             print(f"[PieceProcessor] ⚠️  FILTERED mask {i}: area={full_area:.0f} < threshold={min_area_threshold:.0f} (likely occluded)")
# # #             continue
        
# # #         # Get bounding box of the piece within the board crop
# # #         x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
# # #         if w_p == 0 or h_p == 0:
# # #             continue

# # #         # --- ROBUST BASE DETECTION ---
# # #         # Isolate the bottom 20% of the piece (the "base")
# # #         base_ratio = 0.20
# # #         slice_h = max(1, int(h_p * base_ratio))  # Safety check for small pieces
        
# # #         y_slice_start = y_p + h_p - slice_h
# # #         y_slice_end = y_p + h_p
        
# # #         # Extract the base mask
# # #         base_mask = local_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
        
# # #         # Calculate Centroid of the BASE only
# # #         M = cv2.moments(base_mask)
        
# # #         if M["m00"] > 0:
# # #             cX_base = int(M["m10"] / M["m00"])
# # #             cY_base = int(M["m01"] / M["m00"])
            
# # #             # Map to global coordinates
# # #             pc_global = np.array([x + x_p + cX_base, y + y_slice_start + cY_base])
# # #             area = full_area  # Use full area for deduplication
# # #         else:
# # #             # Fallback: Bottom-center of bounding box
# # #             pc_global = np.array([x + x_p + w_p // 2, y + y_p + h_p])
# # #             area = full_area

# # #         # FILTER: Check if contact point is inside board boundary
# # #         is_inside = cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False)
# # #         if is_inside < 0:  # Point is outside
# # #             continue

# # #         # Find closest tile center
# # #         dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
# # #         closest_idx = np.argmin(dists)
# # #         closest_dist = dists[closest_idx]
        
# # #         piece_data.append({
# # #             'idx': i,
# # #             'mask': full_piece_mask,
# # #             'centroid': pc_global,
# # #             'tile_idx': closest_idx,
# # #             'distance': closest_dist,
# # #             'area': area
# # #         })
    
# # #     if filtered_count > 0:
# # #         print(f"[PieceProcessor] Filtered {filtered_count} partial/occluded masks out of {len(piece_masks_list)}")
    
# # #     # Deduplicate: keep only one piece per tile (largest area)
# # #     tile_to_piece = {}
# # #     for piece in piece_data:
# # #         tile_idx = piece['tile_idx']
# # #         if tile_idx not in tile_to_piece or piece['area'] > tile_to_piece[tile_idx]['area']:
# # #             tile_to_piece[tile_idx] = piece
    
# # #     return tile_to_piece







# # # # """
# # # # Piece detection and assignment logic.
# # # # Handles base-centroid detection and tile assignment with deduplication.
# # # # """

# # # # import cv2
# # # # import numpy as np


# # # # def process_pieces(piece_masks_list, crop_bbox, grid_points_global, tile_centers_global):
# # # #     """
# # # #     Process chess piece masks and assign them to board tiles.
    
# # # #     Args:
# # # #         piece_masks_list: List of binary masks (full image size) for each detected piece
# # # #         crop_bbox: Tuple (x, y, w, h) of the board crop region
# # # #         grid_points_global: Array of 81 grid intersection points (9x9)
# # # #         tile_centers_global: Array of 64 tile center points (8x8)
    
# # # #     Returns:
# # # #         dict: Mapping from tile_idx to piece data dictionary
# # # #     """
# # # #     x, y, w, h = crop_bbox
    
# # # #     # Define board boundary from grid corners
# # # #     grid_corners = np.array([
# # # #         grid_points_global[0],      # top-left (0,0)
# # # #         grid_points_global[8],      # top-right (0,8)
# # # #         grid_points_global[80],     # bottom-right (8,8)
# # # #         grid_points_global[72]      # bottom-left (8,0)
# # # #     ], dtype=np.float32)
    
# # # #     piece_data = []
    
# # # #     for i, full_piece_mask in enumerate(piece_masks_list):
# # # #         # Crop the piece mask to board region
# # # #         local_piece_mask = full_piece_mask[y:y+h, x:x+w]
        
# # # #         # Get bounding box of the piece within the board crop
# # # #         x_p, y_p, w_p, h_p = cv2.boundingRect(local_piece_mask)
# # # #         if w_p == 0 or h_p == 0:
# # # #             continue

# # # #         # --- ROBUST BASE DETECTION ---
# # # #         # Isolate the bottom 20% of the piece (the "base")
# # # #         base_ratio = 0.20
# # # #         slice_h = max(1, int(h_p * base_ratio))  # Safety check for small pieces
        
# # # #         y_slice_start = y_p + h_p - slice_h
# # # #         y_slice_end = y_p + h_p
        
# # # #         # Extract the base mask
# # # #         base_mask = local_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
        
# # # #         # Calculate Centroid of the BASE only
# # # #         M = cv2.moments(base_mask)
        
# # # #         if M["m00"] > 0:
# # # #             cX_base = int(M["m10"] / M["m00"])
# # # #             cY_base = int(M["m01"] / M["m00"])
            
# # # #             # Map to global coordinates
# # # #             pc_global = np.array([x + x_p + cX_base, y + y_slice_start + cY_base])
# # # #             area = cv2.moments(local_piece_mask)["m00"]  # Full area for deduplication
# # # #         else:
# # # #             # Fallback: Bottom-center of bounding box
# # # #             pc_global = np.array([x + x_p + w_p // 2, y + y_p + h_p])
# # # #             area = 0

# # # #         # FILTER: Check if contact point is inside board boundary
# # # #         is_inside = cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False)
# # # #         if is_inside < 0:  # Point is outside
# # # #             continue

# # # #         # Find closest tile center
# # # #         dists = np.linalg.norm(tile_centers_global - pc_global, axis=1)
# # # #         closest_idx = np.argmin(dists)
# # # #         closest_dist = dists[closest_idx]
        
# # # #         piece_data.append({
# # # #             'idx': i,
# # # #             'mask': full_piece_mask,
# # # #             'centroid': pc_global,
# # # #             'tile_idx': closest_idx,
# # # #             'distance': closest_dist,
# # # #             'area': area
# # # #         })
    
# # # #     # Deduplicate: keep only one piece per tile (largest area)
# # # #     tile_to_piece = {}
# # # #     for piece in piece_data:
# # # #         tile_idx = piece['tile_idx']
# # # #         if tile_idx not in tile_to_piece or piece['area'] > tile_to_piece[tile_idx]['area']:
# # # #             tile_to_piece[tile_idx] = piece
    
# # # #     return tile_to_piece
