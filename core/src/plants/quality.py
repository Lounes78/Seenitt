import cv2
import numpy as np
import time

def sharpness_score(pil_img):
    """
    Mesure la netteté via la variance du Laplacien.
    Un score élevé = image nette. Un score faible (< 100) = image floue.
    """
    # Conversion PIL -> OpenCV
    img_gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    
    # Calcul de la variance du Laplacien
    score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return score

def view_quality_score(box, pil_img):
    """Calcul de qualité robuste : Surface x Netteté."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    if w <= 10 or h <= 10: return 0 # Filtre les boîtes invalides
    
    # Crop et conversion pour OpenCV
    crop = pil_img.crop((x1, y1, x2, y2))
    crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    
    # Score = Surface * Variance du Laplacien (netteté)
    sharpness = cv2.Laplacian(crop_cv, cv2.CV_64F).var()
    return (w * h) * sharpness

def frame_difference_score(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(np.array(prev_frame), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(np.array(curr_frame), cv2.COLOR_RGB2GRAY)
    return np.mean(cv2.absdiff(prev_gray, curr_gray))
