"""
Temporal grid tracking using optical flow.
"""

import cv2
import numpy as np


class GridTracker:
    """Tracks chessboard grid points across frames using optical flow."""
    
    def __init__(self):
        self.prev_gray = None
        self.prev_grid_points = None
        self.prev_crop_bbox = None
        self.tracking_enabled = False
    
    def reset(self):
        """Reset tracking state (call after errors or tracking failures)."""
        self.prev_gray = None
        self.prev_grid_points = None
        self.prev_crop_bbox = None
        self.tracking_enabled = False
    
    def try_track(self, curr_gray, curr_crop_bbox):
        """
        Attempt to track grid points from previous frame.
        
        Args:
            curr_gray: Current frame grayscale image (cropped to board)
            curr_crop_bbox: Tuple (x, y, w, h) of current crop region
        
        Returns:
            tuple: (success, grid_points_local, tile_centers_local, stats)
                   If success=False, returns (False, None, None, {})
        """
        if not self.tracking_enabled or self.prev_gray is None:
            return False, None, None, {}
        
        x, y, w, h = curr_crop_bbox
        prev_x, prev_y, prev_w, prev_h = self.prev_crop_bbox
        
        # Check if crop position/size is stable
        dx = abs(x - prev_x)
        dy = abs(y - prev_y)
        dw = abs(w - prev_w)
        dh = abs(h - prev_h)
        
        # Check size match (required for optical flow)
        size_match = (curr_gray.shape == self.prev_gray.shape)
        
        if dx >= 50 or dy >= 50 or dw >= 50 or dh >= 50 or not size_match:
            self.tracking_enabled = False
            return False, None, None, {}
        
        try:
            # FAST: Track grid points using Optical Flow (~2ms)
            tracked_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, self.prev_grid_points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            
            # Strict requirement: need ALL 81 points for reshaping
            num_tracked = np.sum(status)
            
            if num_tracked != 81 or tracked_points.shape[0] != 81:
                self.tracking_enabled = False
                return False, None, None, {}
            
            # Successfully tracked all points
            grid_points_local = tracked_points.reshape(81, 2)
            
            # Reconstruct tile centers from grid (8x8 = 64 centers)
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
            
            return True, grid_points_local, tile_centers_local, {'method': 'optical_flow_tracking'}
            
        except Exception:
            self.tracking_enabled = False
            return False, None, None, {}
    
    def update(self, curr_gray, grid_points_local, curr_crop_bbox):
        """
        Update tracking state after successful grid solve.
        
        Args:
            curr_gray: Current frame grayscale (cropped to board)
            grid_points_local: Solved grid points (81 points)
            curr_crop_bbox: Current crop bounding box
        """
        self.tracking_enabled = True
        self.prev_gray = curr_gray
        self.prev_grid_points = grid_points_local.reshape(-1, 1, 2).astype(np.float32)
        self.prev_crop_bbox = curr_crop_bbox
