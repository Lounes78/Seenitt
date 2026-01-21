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
            # FIX: Return 5 values (None, None, None, False, Stats)
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

        # 2. Hough
        edges = cv2.Canny(img, 100, 550)
        edges_masked = cv2.bitwise_and(edges, edges, mask=segmentation_mask)
        min_dim = min(w, h) if w > 0 and h > 0 else 100
        lines = getHoughLines(edges_masked, min_line_size=0.25 * min_dim)
        
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

            real_intersections = cv2.perspectiveTransform(
                ideal_intersections.reshape(-1, 1, 2), M_inv_final
            ).reshape(-1, 2).astype(np.float32)

            # CANONICALIZE: in order to have same index
            real_intersections = canonicalize_intersections(real_intersections)
            real_tile_centers = centers_from_intersections(real_intersections)

            final_warp_img, _ = getTileImage(
                img_orig, real_intersections[[0, 8, 80, 72]],
                tile_buffer=self.tile_buffer, tile_res=self.tile_res
            )

            return final_warp_img, real_intersections, real_tile_centers, True, stats

        except IndexError:
            stats['error'] = 'Geometry IndexError'
            return None, None, None, False, stats
        except Exception as e:
            stats['error'] = str(e)
            return None, None, None, False, stats



def canonicalize_intersections(real_intersections_81):
    """
    INPUT: (81,2) intersections 9x9 in any orientation
    OUTPUT: (81,2) in canonical order:
        - row 0 is up
        - col 0 is left
    """
    g0 = real_intersections_81.reshape(9, 9, 2)

    cands = []
    g = g0
    for _ in range(4):
        cands.append(g)
        cands.append(np.fliplr(g))
        g = np.rot90(g, 1, axes=(0, 1))

    def score(G):
        # Queremos dx>0 en fila 0, dy>0 en col 0
        dx = np.mean(G[0, 1:, 0] - G[0, :-1, 0])
        dy = np.mean(G[1:, 0, 1] - G[:-1, 0, 1])

        pen = 0.0
        if dx <= 0: pen += 1e6
        if dy <= 0: pen += 1e6

        # preferimos grids "suaves" (monotonos)
        smooth_x = np.mean(np.abs(np.diff(G[0, :, 0])))
        smooth_y = np.mean(np.abs(np.diff(G[:, 0, 1])))
        return pen + 0.001 / (smooth_x + 1e-6) + 0.001 / (smooth_y + 1e-6)

    best = min(cands, key=score)
    return best.reshape(81, 2)

def centers_from_intersections(intersections_81):
    g = intersections_81.reshape(9, 9, 2)
    centers = []
    for r in range(8):
        for c in range(8):
            tl = g[r, c]
            tr = g[r, c+1]
            bl = g[r+1, c]
            br = g[r+1, c+1]
            centers.append((tl + tr + bl + br) / 4.0)
    return np.array(centers, dtype=np.float32)
