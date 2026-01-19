"""
Piece detection and assignment logic.
Handles base-centroid detection and tile assignment with deduplication.
"""

import cv2
import numpy as np


def process_pieces(piece_masks_list, crop_bbox, grid_points_global, tile_centers_global):
    """
    Process chess piece masks and assign them to board tiles.
    
    Args:
        piece_masks_list: List of binary masks (full image size) for each detected piece
        crop_bbox: Tuple (x, y, w, h) of the board crop region
        grid_points_global: Array of 81 grid intersection points (9x9)
        tile_centers_global: Array of 64 tile center points (8x8)
    
    Returns:
        dict: Mapping from tile_idx to piece data dictionary
    """
    x, y, w, h = crop_bbox
    
    # Define board boundary from grid corners
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
        if w_p == 0 or h_p == 0:
            continue

        # --- ROBUST BASE DETECTION ---
        # Isolate the bottom 20% of the piece (the "base")
        base_ratio = 0.20
        slice_h = max(1, int(h_p * base_ratio))  # Safety check for small pieces
        
        y_slice_start = y_p + h_p - slice_h
        y_slice_end = y_p + h_p
        
        # Extract the base mask
        base_mask = local_piece_mask[y_slice_start:y_slice_end, x_p:x_p+w_p]
        
        # Calculate Centroid of the BASE only
        M = cv2.moments(base_mask)
        
        if M["m00"] > 0:
            cX_base = int(M["m10"] / M["m00"])
            cY_base = int(M["m01"] / M["m00"])
            
            # Map to global coordinates
            pc_global = np.array([x + x_p + cX_base, y + y_slice_start + cY_base])
            area = cv2.moments(local_piece_mask)["m00"]  # Full area for deduplication
        else:
            # Fallback: Bottom-center of bounding box
            pc_global = np.array([x + x_p + w_p // 2, y + y_p + h_p])
            area = 0

        # FILTER: Check if contact point is inside board boundary
        is_inside = cv2.pointPolygonTest(grid_corners, tuple(pc_global.astype(float)), False)
        if is_inside < 0:  # Point is outside
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
    
    return tile_to_piece
