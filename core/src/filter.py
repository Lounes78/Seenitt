import numpy as np
import cv2

class QualityFilter:
    def __init__(self, margin=5, min_score=0.5, max_border_contact_ratio=0.15):
        """
        max_border_contact_ratio: 
            The maximum allowed ratio of (pixels touching border) / (total mask perimeter).
            0.15 means if more than 15% of the object's edge is actually the image border, 
            we assume too much is cut off.
        """
        self.margin = margin
        self.min_score = min_score
        self.max_border_contact_ratio = max_border_contact_ratio

    def check(self, mask, score):
        # 1. Score Check
        if score < self.min_score:
            return False, f"Low Score ({score:.2f})"

        # 2. Empty Mask Check
        if np.count_nonzero(mask) == 0:
            return False, "Empty Mask"

        h, w = mask.shape[:2]
        
        # 3. Create a "Border Mask"
        # This mask is 1 only at the image borders (within margin)
        border_mask = np.zeros_like(mask)
        cv2.rectangle(border_mask, (0, 0), (w, h), 255, thickness=self.margin)
        
        # 4. Find the intersection: Where does the chessboard touch the border?
        # The mask is 255 for the object. border_mask is 255 for border.
        # Intersection is where both are 255.
        contact_region = cv2.bitwise_and(mask, border_mask)
        contact_pixels = np.count_nonzero(contact_region)
        
        # 5. Calculate Total Perimeter of the Mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return False, "No Contour"
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0: return False, "Zero Perimeter"

        # 6. Calculate Ratio
        # We roughly approximate perimeter pixels as the arcLength
        contact_ratio = contact_pixels / perimeter

        if contact_ratio > self.max_border_contact_ratio:
            return False, f"Cut Off ({contact_ratio:.1%} > {self.max_border_contact_ratio:.1%})"

        return True, "OK"







# super strict

# import numpy as np
# import cv2

# class QualityFilter:
#     def __init__(self, margin=10, min_score=0.5):
#         self.margin = margin
#         self.min_score = min_score

#     def check(self, mask, score):
#         """
#         Returns: (True, "OK") or (False, "Reason")
#         """
#         # 1. Score Check
#         if score < self.min_score:
#             return False, f"Low Score ({score:.2f})"

#         # 2. Empty Mask Check
#         if np.count_nonzero(mask) == 0:
#             return False, "Empty Mask"

#         # 3. Rigid Visibility Check (BBox must be inside margin)
#         frame_h, frame_w = mask.shape[:2]
#         x, y, w, h = cv2.boundingRect(mask)
#         x2 = x + w
#         y2 = y + h

#         # Check Left and Top
#         if x <= self.margin:
#             return False, "Touches Left Edge"
#         if y <= self.margin:
#             return False, "Touches Top Edge"
            
#         # Check Right and Bottom
#         if x2 >= (frame_w - self.margin):
#             return False, "Touches Right Edge"
#         if y2 >= (frame_h - self.margin):
#             return False, "Touches Bottom Edge"

#         return True, "OK"

# import numpy as np
# import cv2

# class QualityFilter:
#     def __init__(self, min_score=0.5, min_area_ratio=0.05):
#         self.min_score = min_score
#         self.min_area_ratio = min_area_ratio # Mask must be at least 5% of image

#     def check(self, mask, score):
#         """
#         Returns: (True, "OK") or (False, "Reason")
#         """
#         # 1. Score Check
#         if score < self.min_score:
#             return False, f"Low Score ({score:.2f})"

#         # 2. Empty Mask Check
#         if np.count_nonzero(mask) == 0:
#             return False, "Empty Mask"

#         # 3. Get Largest Contour
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             return False, "No Contour Found"
        
#         largest_contour = max(contours, key=cv2.contourArea)

#         # 4. Area Check
#         h, w = mask.shape[:2]
#         image_area = h * w
#         mask_area = cv2.contourArea(largest_contour)
        
#         if (mask_area / image_area) < self.min_area_ratio:
#             return False, "Too Small"

#         # 5. Quadrilateral Check (The new logic)
#         # We simplify the contour to see if it reduces to a 4-sided shape.
#         # epsilon determines how "loose" the approximation is.
#         # 0.04 (4%) is a standard value for finding geometric shapes.
#         perimeter = cv2.arcLength(largest_contour, True)
#         epsilon = 0.04 * perimeter 
#         approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
#         if len(approx) != 4:
#             return False, f"Not a Quad (Vertices: {len(approx)})"

#         # 6. Convexity Check (Optional but recommended)
#         # A chessboard should be convex. If the mask is U-shaped, it's bad.
#         if not cv2.isContourConvex(approx):
#              return False, "Not Convex"

#         return True, "OK"

# import numpy as np
# import cv2

# class QualityFilter:
#     def __init__(self, min_score=0.5, min_area_ratio=0.05, edge_margin=5):
#         self.min_score = min_score
#         self.min_area_ratio = min_area_ratio # Mask must be at least 5% of image
#         self.edge_margin = edge_margin

#     def check(self, mask, score):
#         """
#         Returns: (True, "OK") or (False, "Reason")
#         """
#         # 1. Score Check
#         if score < self.min_score:
#             return False, f"Low Score ({score:.2f})"

#         # 2. Empty Mask Check
#         if np.count_nonzero(mask) == 0:
#             return False, "Empty Mask"

#         # 3. Geometry Checks
#         h, w = mask.shape[:2]
#         x, y, mw, mh = cv2.boundingRect(mask)
        
#         # Area Check
#         mask_area = mw * mh
#         image_area = h * w
#         if (mask_area / image_area) < self.min_area_ratio:
#             return False, "Too Small"

#         # 4. Visibility Check (Refined)
#         # We define limits
#         x2, y2 = x + mw, y + mh
        
#         touches_left = x < self.edge_margin
#         touches_top = y < self.edge_margin
#         touches_right = x2 > (w - self.edge_margin)
#         touches_bottom = y2 > (h - self.edge_margin)
        
#         edges_touched = sum([touches_left, touches_top, touches_right, touches_bottom])
        
#         # If it touches 3 or more sides, it's likely zoomed in too far to be solvable
#         if edges_touched >= 3:
#             return False, "Overly Cropped (>2 edges)"

#         return True, "OK"