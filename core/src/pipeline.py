import cv2
import numpy as np
import os

# Internal imports
from src.segmentation import ChessboardSegmenter
from src.grid_solver import GridSolver
from src.filter import QualityFilter
from src.grid_tracker import GridTracker
from src.piece_processor import process_pieces
from src.visualizer import draw_grid_lines, visualize_pieces

class SAM3Pipeline:
    def __init__(self, assets_dir="./assets"):
        print("Loading SAM3 Engines (Warmup)...")
        self.segmenter = ChessboardSegmenter(assets_dir, prompts=["chessboard", "chess pieces"])
        self.solver = GridSolver()
        self.tracker = GridTracker()
        self.quality_filter = QualityFilter(margin=1, min_score=0.5, max_border_contact_ratio=0.05)
        
        # Warmup with dummy image
        dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
        self.segmenter.encode_image(dummy)
        print("SAM3 Pipeline Ready.")

    def process_frame(self, frame, frame_id):
        """
        Runs full detection pipeline on a single BGR frame.
        Returns: (vis_img, tile_to_piece)
        """
        original_img = frame.copy()
        
        # 1. Vision Encoder
        vision_features, _ = self.segmenter.encode_image(frame)
        
        # 2. Decode Board
        mask, _, score = self.segmenter.decode_from_features(vision_features, frame, "chessboard")
        
        # 3. Filter
        is_good, reason = self.quality_filter.check(mask, score)
        if not is_good:
            cv2.putText(original_img, f"SKIP: {reason}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.tracker.reset()
            return original_img, None

        # 4. Crop Logic
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        crop_img = frame[y:y+h, x:x+w].copy()
        crop_mask = mask[y:y+h, x:x+w]
        curr_crop_bbox = (x, y, w, h)
        curr_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # 5. Tracking / Solving
        success, grid_points_local, tile_centers_local, stats = self.tracker.try_track(
            curr_gray, curr_crop_bbox
        )

        if not success:
            _, grid_points_local, tile_centers_local, success, stats = self.solver.solve(
                crop_img, crop_mask, time_limit_ms=200
            )
            if success:
                self.tracker.update(curr_gray, grid_points_local, curr_crop_bbox)

        # 6. Piece Detection & Visualization
        vis_img = original_img.copy()
        tile_to_piece = {}

        if success:
            # Run Piece Decoder (using cached features)
            piece_masks_list, _, _ = self.segmenter.decode_from_features(vision_features, frame, "chess pieces")
            
            grid_points_global = grid_points_local + np.array([x, y])
            tile_centers_global = tile_centers_local + np.array([x, y])
            
            draw_grid_lines(vis_img, grid_points_global)
            
            # Map pieces to tiles
            tile_to_piece = process_pieces(
                piece_masks_list, curr_crop_bbox, grid_points_global, tile_centers_global
            )
            visualize_pieces(vis_img, tile_to_piece, tile_centers_global)
            
            status_text = "TRACKED" if 'method' in stats and 'tracking' in stats['method'] else "SOLVED"
            color = (0, 255, 0)
        else:
            status_text = "GRID FAILED"
            color = (0, 0, 255)

        cv2.putText(vis_img, f"Frame {frame_id} | {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return vis_img, tile_to_piece
