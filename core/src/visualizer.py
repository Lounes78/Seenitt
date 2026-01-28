"""
Visualization utilities for chessboard detection results.
"""

import cv2
import numpy as np


def draw_grid_lines(img, grid_points_global, color=(255, 0, 0)):
    """
    Draw 9x9 grid lines on the image.

    Args:
        img: Image to draw on (will be modified in-place)
        grid_points_global: Array of 81 grid points in global coordinates
        color: BGR color for grid lines (default: blue)
    """
    grid_9x9 = grid_points_global.reshape(9, 9, 2).astype(np.int32)

    # Draw horizontal lines
    for row in range(9):
        pts = np.ascontiguousarray(grid_9x9[row, :]).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, color, 2)

    # Draw vertical lines
    for col in range(9):
        pts = np.ascontiguousarray(grid_9x9[:, col]).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, color, 2)


def visualize_pieces(img, tile_to_piece, tile_centers_global, color=None):
    """
    Overlay piece masks and draw vectors to assigned tiles.

    Args:
        img: Image to draw on (will be modified in-place)
        tile_to_piece: Dict mapping tile_idx to piece data
        tile_centers_global: Array of tile center points
        color: Optional fixed BGR color for all pieces.
               If None, a unique color is generated per piece.
    """
    for piece in tile_to_piece.values():
        i = piece['idx']
        full_piece_mask = piece['mask']
        pc_global = piece['centroid']
        closest_center = tile_centers_global[piece['tile_idx']]

        # Choose color
        if color is None:
            hue = int((i * 137.508) % 180)
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            piece_color = tuple(map(int, rgb_color))
        else:
            piece_color = color

        # Overlay mask with semi-transparent color
        colored_layer = np.zeros_like(img)
        colored_layer[:] = piece_color

        mask_indices = full_piece_mask > 0
        if np.any(mask_indices):
            img[mask_indices] = cv2.addWeighted(
                img[mask_indices], 0.6,
                colored_layer[mask_indices], 0.4,
                0
            )

        # Draw vector from piece centroid to tile center
        pt1 = tuple(pc_global.astype(int))
        pt2 = tuple(closest_center.astype(int))

        cv2.line(img, pt1, pt2, piece_color, 2)
        cv2.circle(img, pt1, 6, (255, 255, 255), -1)
        cv2.circle(img, pt1, 4, piece_color, -1)




# """
# Visualization utilities for chessboard detection results.
# """

# import cv2
# import numpy as np


# def draw_grid_lines(img, grid_points_global):
#     """
#     Draw 9x9 grid lines on the image.
    
#     Args:
#         img: Image to draw on (will be modified in-place)
#         grid_points_global: Array of 81 grid points in global coordinates
#     """
#     grid_9x9 = grid_points_global.reshape(9, 9, 2).astype(np.int32)
    
#     # Draw horizontal lines
#     for row in range(9):
#         pts = np.ascontiguousarray(grid_9x9[row, :]).reshape((-1, 1, 2))
#         cv2.polylines(img, [pts], False, (255, 0, 0), 2)
    
#     # Draw vertical lines
#     for col in range(9):
#         pts = np.ascontiguousarray(grid_9x9[:, col]).reshape((-1, 1, 2))
#         cv2.polylines(img, [pts], False, (255, 0, 0), 2)


# def visualize_pieces(img, tile_to_piece, tile_centers_global):
#     """
#     Overlay piece masks and draw vectors to assigned tiles.
    
#     Args:
#         img: Image to draw on (will be modified in-place)
#         tile_to_piece: Dict mapping tile_idx to piece data
#         tile_centers_global: Array of tile center points
#     """
#     for piece in tile_to_piece.values():
#         i = piece['idx']
#         full_piece_mask = piece['mask']
#         pc_global = piece['centroid']
#         closest_center = tile_centers_global[piece['tile_idx']]
        
#         # Generate unique color per piece
#         hue = int((i * 137.508) % 180)
#         hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
#         rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
#         piece_color = tuple(map(int, rgb_color))

#         # Overlay mask with semi-transparent color
#         colored_layer = np.zeros_like(img)
#         colored_layer[:] = piece_color

#         mask_indices = full_piece_mask > 0
#         if np.any(mask_indices):
#             img[mask_indices] = cv2.addWeighted(
#                 img[mask_indices], 0.6,
#                 colored_layer[mask_indices], 0.4,
#                 0
#             )

#         # Draw vector from piece centroid to tile center
#         pt1 = tuple(pc_global.astype(int))
#         pt2 = tuple(closest_center.astype(int))
        
#         cv2.line(img, pt1, pt2, piece_color, 2)
#         cv2.circle(img, pt1, 6, (255, 255, 255), -1)
#         cv2.circle(img, pt1, 4, piece_color, -1)
