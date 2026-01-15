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