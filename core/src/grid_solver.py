# import cv2
# import numpy as np
# from time import time

# try:
#     from .geometry.contour_detect import *
#     from .geometry.line_intersection import *
#     from .geometry.rectify_refine import *
# except ImportError:
#     from contour_detect import *
#     from line_intersection import *
#     from rectify_refine import *

# class GridSolver:
#     def __init__(self):
#         self.tile_res = 64
#         self.tile_buffer = 1 

#     def _get_dense_angles_from_crop(self, img_crop, fallback_angle=0):
#         if img_crop.size == 0: return [fallback_angle, (fallback_angle + 90) % 180]

#         # OPTIMIZATION 1: Resize large crops for angle detection
#         # Angles are scale-invariant, so we can work on a 300px thumbnail.
#         h, w = img_crop.shape[:2]
#         scale = 1.0
#         if w > 300:
#             scale = 300.0 / w
#             img_crop = cv2.resize(img_crop, (0, 0), fx=scale, fy=scale)

#         edges = cv2.Canny(img_crop, 100, 550)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

#         contours, _ = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#         contours = simplifyContours(contours)
#         contours, _ = pruneContours(contours)

#         if len(contours) == 0: 
#             return [fallback_angle, (fallback_angle + 90) % 180]

#         thetas = getContourThetas(contours)
#         try:
#             top_two_angles = calculateKDE(thetas)
#         except:
#             return [fallback_angle, (fallback_angle + 90) % 180]
            
#         if hasattr(top_two_angles, 'tolist'):
#             top_two_angles = top_two_angles.tolist()
#         elif not isinstance(top_two_angles, list):
#             top_two_angles = list(top_two_angles)

#         if len(top_two_angles) == 0:
#             return [fallback_angle, (fallback_angle + 90) % 180]
#         elif len(top_two_angles) == 1:
#             angle1 = top_two_angles[0]
#             angle2 = (angle1 + 90) % 180
#             top_two_angles.append(angle2)

#         return top_two_angles

#     def _is_transform_sane(self, M, img_shape):
#         if M is None: return False
#         h, w = img_shape[:2]
#         corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
#         try:
#             dst = cv2.perspectiveTransform(corners, M)
#             x_min, y_min = np.min(dst, axis=0)[0]
#             x_max, y_max = np.max(dst, axis=0)[0]
#             # Sane checks: Not huge (4x size) and not tiny (<10px)
#             if (x_max - x_min) > w * 4 or (y_max - y_min) > h * 4: return False
#             if (x_max - x_min) < 10 or (y_max - y_min) < 10: return False
#         except Exception:
#             return False
#         return True

#     def solve(self, img, segmentation_mask, time_limit_ms=200):
#         stats = {}
#         t_start = time()

#         def check_timeout(step_name):
#             if (time() - t_start) * 1000 > time_limit_ms:
#                 stats['timeout_at'] = step_name
#                 return True
#             return False

#         if segmentation_mask.dtype != np.uint8:
#             segmentation_mask = segmentation_mask.astype(np.uint8)

#         img_orig = img.copy()
#         h_img, w_img = img.shape[:2]

#         # 1. Geom Analysis
#         contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours: 
#             return None, None, None, False, {'error': 'No mask'}
        
#         largest_contour = max(contours, key=cv2.contourArea)
#         rect = cv2.minAreaRect(largest_contour)
#         mask_angle = rect[2]
#         if mask_angle < -45: mask_angle += 90 
        
#         simplified_contour = simplifyContours([largest_contour])[0]
#         x, y, w, h = cv2.boundingRect(largest_contour)
        
#         if len(simplified_contour) < 4:
#             simplified_contour = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
        
#         pad = 20
#         y1 = max(0, y - pad); y2 = min(h_img, y + h + pad)
#         x1 = max(0, x - pad); x2 = min(w_img, x + w + pad)
#         img_crop = img[y1:y2, x1:x2]
        
#         # Fast Angle Detection (using optimization inside the function)
#         top_two_angles = self._get_dense_angles_from_crop(img_crop, fallback_angle=mask_angle)
        
#         if check_timeout('geom_analysis'): return None, None, None, False, stats

#         # 2. Hough (OPTIMIZED: Run only on Crop)
#         # ---------------------------------------------------------
#         # A. Crop inputs first
#         roi_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
#         roi_mask = segmentation_mask[y1:y2, x1:x2]

#         # B. Apply CLAHE & Canny on ROI only (Fast!)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced_gray = clahe.apply(roi_gray)
#         edges = cv2.Canny(enhanced_gray, 50, 200)
        
#         edges_masked = cv2.bitwise_and(edges, edges, mask=roi_mask)
#         min_dim = min(w, h) if w > 0 and h > 0 else 100
        
#         # C. Hough Lines on ROI
#         lines = getHoughLines(edges_masked, min_line_size=0.1 * min_dim)

#         # D. Translate lines back to Global Coordinates
#         if lines is not None:
#             # Hough returns [x1, y1, x2, y2]
#             lines[:, [0, 2]] += x1
#             lines[:, [1, 3]] += y1
#         # ---------------------------------------------------------
        
#         if check_timeout('hough'): return None, None, None, False, stats

#         lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=35)
        
#         if len(lines_a) < 2 or len(lines_b) < 2:
#             stats['error'] = 'Not enough lines'
#             return None, None, None, False, stats

#         # 3. RANSAC (OPTIMIZED: Reduced Iterations)
#         best_M = None
#         best_lines_x = []
#         best_lines_y = []
#         step_x = step_y = 0
#         found_match = False
        
#         # Reduced Outer loop 5 -> 2, Inner 50 -> 20. 
#         # Since we have a good mask, probability of finding the grid is high.
#         for i_out in range(2): 
#             if found_match: break
#             if check_timeout('ransac_outer'): return None, None, None, False, stats
            
#             for i_in in range(20):
#                 if check_timeout('ransac_inner'): return None, None, None, False, stats

#                 corners = chooseRandomGoodQuad(lines_a, lines_b, simplified_contour)
#                 if len(corners) != 4: continue

#                 M = getTileTransform(corners.astype(np.float32), tile_buffer=16, tile_res=16)
                
#                 # ... [Existing RANSAC Verification Logic] ...
#                 all_lines = np.vstack([lines_a[:,:2], lines_a[:,2:], lines_b[:,:2], lines_b[:,2:]]).astype(np.float32)
#                 warp_pts = cv2.perspectiveTransform(all_lines[None,:,:], M)[0,:,:]
                
#                 split_idx = 2 * len(lines_a)
#                 warp_lines_a = np.hstack([warp_pts[:len(lines_a)], warp_pts[len(lines_a):split_idx]])
#                 warp_lines_b = np.hstack([warp_pts[split_idx:split_idx+len(lines_b)], warp_pts[split_idx+len(lines_b):]])

#                 thetas_a = np.array([getSegmentTheta(line) for line in warp_lines_a])
#                 thetas_b = np.array([getSegmentTheta(line) for line in warp_lines_b])
                
#                 if len(thetas_a) == 0 or len(thetas_b) == 0: continue

#                 med_a = np.median(thetas_a * 180 / np.pi)
#                 med_b = np.median(thetas_b * 180 / np.pi)
                
#                 if (angleCloseDeg(abs(med_a), 0, 2.0) and angleCloseDeg(abs(med_b), 90, 2.0)) or \
#                    (angleCloseDeg(abs(med_a), 90, 2.0) and angleCloseDeg(abs(med_b), 0, 2.0)):
                    
#                     if not self._is_transform_sane(M, img_orig.shape): continue

#                     warp_img_check, _ = getTileImage(img_orig, corners.astype(np.float32), tile_buffer=16, tile_res=16)
#                     lx, ly, sx, sy = getWarpCheckerLines(warp_img_check)
                    
#                     if len(lx) > 0:
#                         best_lines_x, best_lines_y = lx, ly
#                         step_x, step_y = sx, sy
#                         best_M = M
#                         found_match = True
#                         break
            
#         if not found_match:
#             stats['error'] = 'RANSAC failed'
#             return None, None, None, False, stats

#         if (time_limit_ms - ((time() - t_start) * 1000)) < 30.0: # Tightened margin
#              stats['timeout_at'] = 'safety_margin_pre_refine'
#              return None, None, None, False, stats

#         # 4. Refinement
#         warp_corners, all_warp_corners = getRectChessCorners(best_lines_x, best_lines_y)
#         tile_centers = all_warp_corners + np.array([step_x/2.0, step_y/2.0])
        
#         M_inv = np.matrix(np.linalg.inv(best_M))
#         real_corners, _ = getOrigChessCorners(warp_corners, tile_centers, M_inv)
        
#         warp_img, better_M = getTileImage(img_orig, real_corners, tile_buffer=self.tile_buffer, tile_res=self.tile_res)
#         warp_img, _, refine_M = reRectifyImages(warp_img)
        
#         combined_M = np.matmul(refine_M, better_M)
#         M_inv_final = np.matrix(np.linalg.inv(combined_M))
        
#         # 5. Generate Grid Points
#         h_grid = (np.arange(9) + self.tile_buffer) * self.tile_res
#         v_grid = (np.arange(9) + self.tile_buffer) * self.tile_res
#         xx_grid, yy_grid = np.meshgrid(h_grid, v_grid)
#         ideal_intersections = np.vstack([xx_grid.flatten(), yy_grid.flatten()]).T.astype(np.float32)

#         h_centers = (np.arange(8) + 0.5 + self.tile_buffer) * self.tile_res
#         v_centers = (np.arange(8) + 0.5 + self.tile_buffer) * self.tile_res
#         xx_cent, yy_cent = np.meshgrid(h_centers, v_centers)
#         ideal_centers = np.vstack([xx_cent.flatten(), yy_cent.flatten()]).T.astype(np.float32)

#         real_intersections = cv2.perspectiveTransform(ideal_intersections.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
#         real_tile_centers = cv2.perspectiveTransform(ideal_centers.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
        
#         # === FIX START: Systematic Offset Correction ===
#         final_warp_img, _ = getTileImage(img_orig, real_intersections[[0, 8, 80, 72]], tile_buffer=self.tile_buffer, tile_res=self.tile_res)
        
#         if final_warp_img is not None and final_warp_img.size > 0:
#             gray_warp = cv2.cvtColor(final_warp_img, cv2.COLOR_BGR2GRAY)
#             r_h = self.tile_res
            
#             var_top = np.var(gray_warp[0:r_h, :])
#             var_bot = np.var(gray_warp[7*r_h:8*r_h, :])
            
#             shift_y = 0
#             if var_bot < (var_top / 3.0): shift_y = -1 
#             elif var_top < (var_bot / 3.0): shift_y = 1
            
#             if shift_y != 0:
#                 shift_px = shift_y * self.tile_res
#                 ideal_intersections[:, 1] += shift_px
#                 ideal_centers[:, 1] += shift_px
                
#                 real_intersections = cv2.perspectiveTransform(ideal_intersections.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
#                 real_tile_centers = cv2.perspectiveTransform(ideal_centers.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
                
#                 final_warp_img, _ = getTileImage(img_orig, real_intersections[[0, 8, 80, 72]], tile_buffer=self.tile_buffer, tile_res=self.tile_res)
#         # === FIX END ===

#         return final_warp_img, real_intersections, real_tile_centers, True, stats















import cv2
import numpy as np
from time import time

try:
    from .geometry.contour_detect import *
    from .geometry.line_intersection import *
    from .geometry.rectify_refine import *
except ImportError:
    from contour_detect import *
    from line_intersection import *
    from rectify_refine import *

class GridSolver:
    def __init__(self):
        self.tile_res = 64
        self.tile_buffer = 1 



    def _correct_grid_offset_with_mask(self, M_inv, mask, ideal_intersections, ideal_centers):
        """
        Shifts the grid logic (up/down/left/right) and checks which alignment 
        has the highest overlap with the segmentation mask.
        """
        h_img, w_img = mask.shape[:2]
        
        # 1. Define candidates: (dx, dy) in 'tile' units
        # (0,0) = No shift
        # (0, 1) = Grid needs to move DOWN (was detected too high)
        # (0, -1) = Grid needs to move UP (was detected too low)
        candidates = [
            (0, 0),   # Original
            (0, 1),   # Shift Down
            (0, -1),  # Shift Up
            (1, 0),   # Shift Right
            (-1, 0)   # Shift Left
        ]

        best_score = -1
        best_offset = (0, 0)
        
        # We only need the 4 outer corners to draw the polygon
        # Indices for 9x9 grid: TopLeft=0, TopRight=8, BotRight=80, BotLeft=72
        corner_indices = [0, 8, 80, 72]

        for dx, dy in candidates:
            # Shift the LOGICAL coordinates (not pixels)
            # We add (dx * tile_res) to the ideal coordinates
            shift_x = dx * self.tile_res
            shift_y = dy * self.tile_res
            
            shifted_ideal = ideal_intersections.copy()
            shifted_ideal[:, 0] += shift_x
            shifted_ideal[:, 1] += shift_y
            
            # Project to pixel space
            # shape needs to be (N, 1, 2) for perspectiveTransform
            projected_corners = cv2.perspectiveTransform(
                shifted_ideal[corner_indices].reshape(-1, 1, 2), 
                M_inv
            ).reshape(-1, 2).astype(np.int32)

            # Create a "Grid Mask" for this candidate
            candidate_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(candidate_mask, projected_corners, 255)

            # Calculate Overlap
            # bitwise_and finds pixels where BOTH the candidate grid AND the AI mask are white
            overlap = cv2.bitwise_and(mask, candidate_mask)
            overlap_count = np.count_nonzero(overlap)
            
            # Optional: Normalize by the candidate area to prevent 
            # favouring purely larger grids (though perspective makes them similar)
            candidate_area = np.count_nonzero(candidate_mask)
            if candidate_area == 0: continue
            
            score = overlap_count / candidate_area

            if score > best_score:
                best_score = score
                best_offset = (dx, dy)

        # Apply the winner
        if best_offset != (0, 0):
            # print(f"Correction Applied: {best_offset}") # Debug print
            sx = best_offset[0] * self.tile_res
            sy = best_offset[1] * self.tile_res
            
            # Permanently shift the ideal points
            ideal_intersections[:, 0] += sx
            ideal_intersections[:, 1] += sy
            ideal_centers[:, 0] += sx
            ideal_centers[:, 1] += sy

        return ideal_intersections, ideal_centers
    

    def _get_dense_angles_from_crop(self, img_crop, fallback_angle=0):
        if img_crop.size == 0: return [fallback_angle, (fallback_angle + 90) % 180]

        edges = cv2.Canny(img_crop, 100, 550)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

        contours, _ = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = simplifyContours(contours)
        contours, _ = pruneContours(contours)

        if len(contours) == 0: 
            return [fallback_angle, (fallback_angle + 90) % 180]

        thetas = getContourThetas(contours)
        try:
            top_two_angles = calculateKDE(thetas)
        except:
            return [fallback_angle, (fallback_angle + 90) % 180]
            
        if hasattr(top_two_angles, 'tolist'):
            top_two_angles = top_two_angles.tolist()
        elif not isinstance(top_two_angles, list):
            top_two_angles = list(top_two_angles)

        if len(top_two_angles) == 0:
            return [fallback_angle, (fallback_angle + 90) % 180]
        elif len(top_two_angles) == 1:
            angle1 = top_two_angles[0]
            angle2 = (angle1 + 90) % 180
            top_two_angles.append(angle2)

        return top_two_angles

    def _is_transform_sane(self, M, img_shape):
        if M is None: return False
        h, w = img_shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            dst = cv2.perspectiveTransform(corners, M)
            x_min, y_min = np.min(dst, axis=0)[0]
            x_max, y_max = np.max(dst, axis=0)[0]
            if (x_max - x_min) > w * 4 or (y_max - y_min) > h * 4: return False
            if (x_max - x_min) < 10 or (y_max - y_min) < 10: return False
        except Exception:
            return False
        return True

    def solve(self, img, segmentation_mask, time_limit_ms=200):
        stats = {}
        t_start = time()

        def check_timeout(step_name):
            if (time() - t_start) * 1000 > time_limit_ms:
                stats['timeout_at'] = step_name
                return True
            return False

        if segmentation_mask.dtype != np.uint8:
            segmentation_mask = segmentation_mask.astype(np.uint8)

        img_orig = img.copy()

        # 1. Geom Analysis
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            return None, None, None, False, {'error': 'No mask'}
        
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        mask_angle = rect[2]
        if mask_angle < -45: mask_angle += 90 
        
        simplified_contour = simplifyContours([largest_contour])[0]
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if len(simplified_contour) < 4:
            simplified_contour = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
        
        pad = 20
        h_img, w_img = img.shape[:2]
        y1 = max(0, y - pad); y2 = min(h_img, y + h + pad)
        x1 = max(0, x - pad); x2 = min(w_img, x + w + pad)
        img_crop = img[y1:y2, x1:x2]
        
        top_two_angles = self._get_dense_angles_from_crop(img_crop, fallback_angle=mask_angle)
        
        if check_timeout('geom_analysis'): return None, None, None, False, stats

        # # 2. Hough
        # edges = cv2.Canny(img, 100, 550)
        # edges_masked = cv2.bitwise_and(edges, edges, mask=segmentation_mask)
        # min_dim = min(w, h) if w > 0 and h > 0 else 100
        # lines = getHoughLines(edges_masked, min_line_size=0.25 * min_dim)
        
        # if check_timeout('hough'): return None, None, None, False, stats

        # lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=35)
        
        # if len(lines_a) < 2 or len(lines_b) < 2:
        #     stats['error'] = 'Not enough lines'
        #     return None, None, None, False, stats



        # 2. Hough (UPDATED FOR GLARE HANDLING)
        # ---------------------------------------------------------
        # A. Convert to Grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # B. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This evens out the lighting, making lines visible even in bright spots.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray_img)

        # C. Canny Edge Detection
        # We lower the thresholds (from 100/550 to 50/200) to catch lines 
        # that are washed out by the glare.
        edges = cv2.Canny(enhanced_gray, 50, 200)
        
        edges_masked = cv2.bitwise_and(edges, edges, mask=segmentation_mask)
        min_dim = min(w, h) if w > 0 and h > 0 else 100
        
        # D. Hough Lines
        # CHANGED: min_line_size from 0.25 -> 0.1 (10% of board width)
        # This allows the code to accept the "broken" line segments caused by glare.
        lines = getHoughLines(edges_masked, min_line_size=0.1 * min_dim)
        # ---------------------------------------------------------
        
        if check_timeout('hough'): return None, None, None, False, stats

        lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=35)
        
        if len(lines_a) < 2 or len(lines_b) < 2:
            stats['error'] = 'Not enough lines'
            return None, None, None, False, stats
            

        # 3. RANSAC
        best_M = None
        best_lines_x = []
        best_lines_y = []
        step_x = step_y = 0
        found_match = False
        
        try:
            for i_out in range(5): 
                if found_match: break
                if check_timeout('ransac_outer'): return None, None, None, False, stats
                
                for i_in in range(50):
                    if check_timeout('ransac_inner'): return None, None, None, False, stats

                    corners = chooseRandomGoodQuad(lines_a, lines_b, simplified_contour)
                    if len(corners) != 4: continue

                    M = getTileTransform(corners.astype(np.float32), tile_buffer=16, tile_res=16)
                    all_lines = np.vstack([lines_a[:,:2], lines_a[:,2:], lines_b[:,:2], lines_b[:,2:]]).astype(np.float32)
                    warp_pts = cv2.perspectiveTransform(all_lines[None,:,:], M)[0,:,:]
                    
                    split_idx = 2 * len(lines_a)
                    warp_lines_a = np.hstack([warp_pts[:len(lines_a)], warp_pts[len(lines_a):split_idx]])
                    warp_lines_b = np.hstack([warp_pts[split_idx:split_idx+len(lines_b)], warp_pts[split_idx+len(lines_b):]])

                    thetas_a = np.array([getSegmentTheta(line) for line in warp_lines_a])
                    thetas_b = np.array([getSegmentTheta(line) for line in warp_lines_b])
                    
                    med_a = np.median(thetas_a * 180 / np.pi)
                    med_b = np.median(thetas_b * 180 / np.pi)
                    
                    if (angleCloseDeg(abs(med_a), 0, 2.0) and angleCloseDeg(abs(med_b), 90, 2.0)) or \
                       (angleCloseDeg(abs(med_a), 90, 2.0) and angleCloseDeg(abs(med_b), 0, 2.0)):
                        
                        if not self._is_transform_sane(M, img_orig.shape): continue

                        warp_img_check, _ = getTileImage(img_orig, corners.astype(np.float32), tile_buffer=16, tile_res=16)
                        lx, ly, sx, sy = getWarpCheckerLines(warp_img_check)
                        
                        if len(lx) > 0:
                            best_lines_x, best_lines_y = lx, ly
                            step_x, step_y = sx, sy
                            best_M = M
                            found_match = True
                            break
            
            if not found_match:
                stats['error'] = 'RANSAC failed'
                return None, None, None, False, stats

            if (time_limit_ms - ((time() - t_start) * 1000)) < 50.0:
                 stats['timeout_at'] = 'safety_margin_pre_refine'
                 return None, None, None, False, stats

            # 4. Refinement
            warp_corners, all_warp_corners = getRectChessCorners(best_lines_x, best_lines_y)
            tile_centers = all_warp_corners + np.array([step_x/2.0, step_y/2.0])
            
            M_inv = np.matrix(np.linalg.inv(best_M))
            real_corners, _ = getOrigChessCorners(warp_corners, tile_centers, M_inv)
            
            warp_img, better_M = getTileImage(img_orig, real_corners, tile_buffer=self.tile_buffer, tile_res=self.tile_res)
            warp_img, _, refine_M = reRectifyImages(warp_img)
            
            combined_M = np.matmul(refine_M, better_M)
            M_inv_final = np.matrix(np.linalg.inv(combined_M))
            
            # 5. Generate Grid Points
            h_grid = (np.arange(9) + self.tile_buffer) * self.tile_res
            v_grid = (np.arange(9) + self.tile_buffer) * self.tile_res
            xx_grid, yy_grid = np.meshgrid(h_grid, v_grid)
            ideal_intersections = np.vstack([xx_grid.flatten(), yy_grid.flatten()]).T.astype(np.float32)

            h_centers = (np.arange(8) + 0.5 + self.tile_buffer) * self.tile_res
            v_centers = (np.arange(8) + 0.5 + self.tile_buffer) * self.tile_res
            xx_cent, yy_cent = np.meshgrid(h_centers, v_centers)
            ideal_centers = np.vstack([xx_cent.flatten(), yy_cent.flatten()]).T.astype(np.float32)

            real_intersections = cv2.perspectiveTransform(ideal_intersections.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
            real_tile_centers = cv2.perspectiveTransform(ideal_centers.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
            
            final_warp_img, _ = getTileImage(img_orig, real_intersections[[0, 8, 80, 72]], tile_buffer=self.tile_buffer, tile_res=self.tile_res)

            # === FIX START: Systematic Offset Correction (Variance Heuristic) ===
            # Detect if grid is shifted by checking if the top or bottom row looks "empty" (low variance)
            if final_warp_img is not None and final_warp_img.size > 0:
                gray_warp = cv2.cvtColor(final_warp_img, cv2.COLOR_BGR2GRAY)
                r_h = self.tile_res
                
                # Compare variance of top row vs bottom row
                # High variance = Checkers/Pieces. Low variance = Table/Border.
                var_top = np.var(gray_warp[0:r_h, :])
                var_bot = np.var(gray_warp[7*r_h:8*r_h, :])
                
                shift_y = 0
                # If bottom is very "boring" compared to top, we are shifted down. Move UP.
                if var_bot < (var_top / 3.0):
                    shift_y = -1 
                # If top is very "boring" compared to bottom, we are shifted up. Move DOWN.
                elif var_top < (var_bot / 3.0):
                    shift_y = 1
                
                if shift_y != 0:
                    shift_px = shift_y * self.tile_res
                    ideal_intersections[:, 1] += shift_px
                    ideal_centers[:, 1] += shift_px
                    
                    # Re-project with corrected coordinates
                    real_intersections = cv2.perspectiveTransform(ideal_intersections.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
                    real_tile_centers = cv2.perspectiveTransform(ideal_centers.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
                    
                    # Optional: update warp image for display
                    final_warp_img, _ = getTileImage(img_orig, real_intersections[[0, 8, 80, 72]], tile_buffer=self.tile_buffer, tile_res=self.tile_res)
            # === FIX END ===

            return final_warp_img, real_intersections, real_tile_centers, True, stats

        except IndexError:
            stats['error'] = 'Geometry IndexError'
            return None, None, None, False, stats
        except Exception as e:
            stats['error'] = str(e)
            return None, None, None, False, stats

















# # import cv2
# # import numpy as np
# # from time import time

# # try:
# #     from .geometry.contour_detect import *
# #     from .geometry.line_intersection import *
# #     from .geometry.rectify_refine import *
# # except ImportError:
# #     from contour_detect import *
# #     from line_intersection import *
# #     from rectify_refine import *

# # class GridSolver:
# #     def __init__(self):
# #         self.tile_res = 64
# #         self.tile_buffer = 1 

# #     def _get_dense_angles_from_crop(self, img_crop, fallback_angle=0):
# #         if img_crop.size == 0: return [fallback_angle, (fallback_angle + 90) % 180]

# #         edges = cv2.Canny(img_crop, 100, 550)
# #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# #         edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

# #         contours, _ = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
# #         contours = simplifyContours(contours)
# #         contours, _ = pruneContours(contours)

# #         if len(contours) == 0: 
# #             return [fallback_angle, (fallback_angle + 90) % 180]

# #         thetas = getContourThetas(contours)
# #         try:
# #             top_two_angles = calculateKDE(thetas)
# #         except:
# #             return [fallback_angle, (fallback_angle + 90) % 180]
            
# #         if hasattr(top_two_angles, 'tolist'):
# #             top_two_angles = top_two_angles.tolist()
# #         elif not isinstance(top_two_angles, list):
# #             top_two_angles = list(top_two_angles)

# #         if len(top_two_angles) == 0:
# #             return [fallback_angle, (fallback_angle + 90) % 180]
# #         elif len(top_two_angles) == 1:
# #             angle1 = top_two_angles[0]
# #             angle2 = (angle1 + 90) % 180
# #             top_two_angles.append(angle2)

# #         return top_two_angles

# #     def _is_transform_sane(self, M, img_shape):
# #         if M is None: return False
# #         h, w = img_shape[:2]
# #         corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
# #         try:
# #             dst = cv2.perspectiveTransform(corners, M)
# #             x_min, y_min = np.min(dst, axis=0)[0]
# #             x_max, y_max = np.max(dst, axis=0)[0]
# #             if (x_max - x_min) > w * 4 or (y_max - y_min) > h * 4: return False
# #             if (x_max - x_min) < 10 or (y_max - y_min) < 10: return False
# #         except Exception:
# #             return False
# #         return True

# #     def solve(self, img, segmentation_mask, time_limit_ms=200):
# #         stats = {}
# #         t_start = time()

# #         def check_timeout(step_name):
# #             if (time() - t_start) * 1000 > time_limit_ms:
# #                 stats['timeout_at'] = step_name
# #                 return True
# #             return False

# #         if segmentation_mask.dtype != np.uint8:
# #             segmentation_mask = segmentation_mask.astype(np.uint8)

# #         img_orig = img.copy()

# #         # 1. Geom Analysis
# #         contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #         if not contours: 
# #             # FIX: Return 5 values (None, None, None, False, Stats)
# #             return None, None, None, False, {'error': 'No mask'}
        
# #         largest_contour = max(contours, key=cv2.contourArea)
# #         rect = cv2.minAreaRect(largest_contour)
# #         mask_angle = rect[2]
# #         if mask_angle < -45: mask_angle += 90 
        
# #         simplified_contour = simplifyContours([largest_contour])[0]
# #         x, y, w, h = cv2.boundingRect(largest_contour)
        
# #         if len(simplified_contour) < 4:
# #             simplified_contour = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
        
# #         pad = 20
# #         h_img, w_img = img.shape[:2]
# #         y1 = max(0, y - pad); y2 = min(h_img, y + h + pad)
# #         x1 = max(0, x - pad); x2 = min(w_img, x + w + pad)
# #         img_crop = img[y1:y2, x1:x2]
        
# #         top_two_angles = self._get_dense_angles_from_crop(img_crop, fallback_angle=mask_angle)
        
# #         if check_timeout('geom_analysis'): return None, None, None, False, stats

# #         # 2. Hough
# #         edges = cv2.Canny(img, 100, 550)
# #         edges_masked = cv2.bitwise_and(edges, edges, mask=segmentation_mask)
# #         min_dim = min(w, h) if w > 0 and h > 0 else 100
# #         lines = getHoughLines(edges_masked, min_line_size=0.25 * min_dim)
        
# #         if check_timeout('hough'): return None, None, None, False, stats

# #         lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=35)
        
# #         if len(lines_a) < 2 or len(lines_b) < 2:
# #             stats['error'] = 'Not enough lines'
# #             return None, None, None, False, stats

# #         # 3. RANSAC
# #         best_M = None
# #         best_lines_x = []
# #         best_lines_y = []
# #         step_x = step_y = 0
# #         found_match = False
        
# #         try:
# #             for i_out in range(5): 
# #                 if found_match: break
# #                 if check_timeout('ransac_outer'): return None, None, None, False, stats
                
# #                 for i_in in range(50):
# #                     if check_timeout('ransac_inner'): return None, None, None, False, stats

# #                     corners = chooseRandomGoodQuad(lines_a, lines_b, simplified_contour)
# #                     if len(corners) != 4: continue

# #                     M = getTileTransform(corners.astype(np.float32), tile_buffer=16, tile_res=16)
# #                     all_lines = np.vstack([lines_a[:,:2], lines_a[:,2:], lines_b[:,:2], lines_b[:,2:]]).astype(np.float32)
# #                     warp_pts = cv2.perspectiveTransform(all_lines[None,:,:], M)[0,:,:]
                    
# #                     split_idx = 2 * len(lines_a)
# #                     warp_lines_a = np.hstack([warp_pts[:len(lines_a)], warp_pts[len(lines_a):split_idx]])
# #                     warp_lines_b = np.hstack([warp_pts[split_idx:split_idx+len(lines_b)], warp_pts[split_idx+len(lines_b):]])

# #                     thetas_a = np.array([getSegmentTheta(line) for line in warp_lines_a])
# #                     thetas_b = np.array([getSegmentTheta(line) for line in warp_lines_b])
                    
# #                     med_a = np.median(thetas_a * 180 / np.pi)
# #                     med_b = np.median(thetas_b * 180 / np.pi)
                    
# #                     if (angleCloseDeg(abs(med_a), 0, 2.0) and angleCloseDeg(abs(med_b), 90, 2.0)) or \
# #                        (angleCloseDeg(abs(med_a), 90, 2.0) and angleCloseDeg(abs(med_b), 0, 2.0)):
                        
# #                         if not self._is_transform_sane(M, img_orig.shape): continue

# #                         warp_img_check, _ = getTileImage(img_orig, corners.astype(np.float32), tile_buffer=16, tile_res=16)
# #                         lx, ly, sx, sy = getWarpCheckerLines(warp_img_check)
                        
# #                         if len(lx) > 0:
# #                             best_lines_x, best_lines_y = lx, ly
# #                             step_x, step_y = sx, sy
# #                             best_M = M
# #                             found_match = True
# #                             break
            
# #             if not found_match:
# #                 stats['error'] = 'RANSAC failed'
# #                 return None, None, None, False, stats

# #             if (time_limit_ms - ((time() - t_start) * 1000)) < 50.0:
# #                  stats['timeout_at'] = 'safety_margin_pre_refine'
# #                  return None, None, None, False, stats

# #             # 4. Refinement
# #             warp_corners, all_warp_corners = getRectChessCorners(best_lines_x, best_lines_y)
# #             tile_centers = all_warp_corners + np.array([step_x/2.0, step_y/2.0])
            
# #             M_inv = np.matrix(np.linalg.inv(best_M))
# #             real_corners, _ = getOrigChessCorners(warp_corners, tile_centers, M_inv)
            
# #             warp_img, better_M = getTileImage(img_orig, real_corners, tile_buffer=self.tile_buffer, tile_res=self.tile_res)
# #             warp_img, _, refine_M = reRectifyImages(warp_img)
            
# #             combined_M = np.matmul(refine_M, better_M)
# #             M_inv_final = np.matrix(np.linalg.inv(combined_M))
            
# #             # 5. Generate Grid Points
# #             h_grid = (np.arange(9) + self.tile_buffer) * self.tile_res
# #             v_grid = (np.arange(9) + self.tile_buffer) * self.tile_res
# #             xx_grid, yy_grid = np.meshgrid(h_grid, v_grid)
# #             ideal_intersections = np.vstack([xx_grid.flatten(), yy_grid.flatten()]).T.astype(np.float32)

# #             h_centers = (np.arange(8) + 0.5 + self.tile_buffer) * self.tile_res
# #             v_centers = (np.arange(8) + 0.5 + self.tile_buffer) * self.tile_res
# #             xx_cent, yy_cent = np.meshgrid(h_centers, v_centers)
# #             ideal_centers = np.vstack([xx_cent.flatten(), yy_cent.flatten()]).T.astype(np.float32)

# #             real_intersections = cv2.perspectiveTransform(ideal_intersections.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
# #             real_tile_centers = cv2.perspectiveTransform(ideal_centers.reshape(-1, 1, 2), M_inv_final).reshape(-1, 2)
            
# #             final_warp_img, _ = getTileImage(img_orig, real_intersections[[0, 8, 80, 72]], tile_buffer=self.tile_buffer, tile_res=self.tile_res)

# #             return final_warp_img, real_intersections, real_tile_centers, True, stats

# #         except IndexError:
# #             stats['error'] = 'Geometry IndexError'
# #             return None, None, None, False, stats
# #         except Exception as e:
# #             stats['error'] = str(e)
# #             return None, None, None, False, stats





