import torch
import cv2
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# ---------------------------------------------
# HELPER FUNCTIONS FOR GEOMETRIC RECTIFICATION
# ---------------------------------------------
def order_points(pts):
    """Sorts points: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect



def get_board_corners(mask_binary):
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return None
    
    # Get the largest contour (the board)
    largest = max(contours, key=cv2.contourArea)
    
    # 1. Smooth the shape using Convex Hull
    hull = cv2.convexHull(largest)
    perimeter = cv2.arcLength(hull, True)
    
    # 2. Adaptive approximation: Try to find 4 corners
    for eps_factor in np.linspace(0.02, 0.1, 10):
        epsilon = eps_factor * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            return approx.reshape(4, 2)
            
    # 3. Fallback: Rotated Bounding Box
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.int32(box) 
    return box

def get_birdseye_view(img, corners, size=640, margin_ratio=0.15):
    """
    Warps the image to a top-down view with padding to keep piece heads visible.
    margin_ratio: Percentage of image to leave as margin (0.15 = 15%)
    """
    rect = order_points(corners)
    
    # Calculate margin in pixels
    margin = int(size * margin_ratio)
    
    # Define destination points shifted INWARD by the margin
    dst = np.array([
        [margin, margin],
        [size - margin - 1, margin],
        [size - margin - 1, size - margin - 1],
        [margin, size - margin - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, M

def draw_grid_on_warped(warped_img, size=640, margin_ratio=0.15):
    """Draws the 8x8 grid on the warped image to verify alignment."""
    vis = warped_img.copy()
    margin = int(size * margin_ratio)
    board_span = size - (2 * margin)
    square_size = board_span / 8
    
    # Draw vertical lines
    for i in range(9):
        x = int(margin + i * square_size)
        cv2.line(vis, (x, margin), (x, size - margin), (0, 255, 0), 2)
        
    # Draw horizontal lines
    for i in range(9):
        y = int(margin + i * square_size)
        cv2.line(vis, (margin, y), (size - margin, y), (0, 255, 0), 2)
        
    return vis

def update_board_state_sam3(warped_img, image_model, board_size=640):
    """
    Detects pieces using SAM3, finds their 'feet', and maps them to the 8x8 grid.
    Returns: 8x8 Numpy array (0=Empty, 1=White, 2=Black)
    """
    # Prepare image
    img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Initialize empty 8x8 board
    board_state = np.zeros((8, 8), dtype=int)
    square_size = board_size / 8
    
    # --- ADDED: Draw grid overlay first so it is underneath detections ---
    vis_img = draw_grid_on_warped(warped_img, size=board_size, margin_ratio=0.15)
    
    # Collect all detections
    all_boxes = []
    all_scores = []
    all_colors = []  # 1 for white, 2 for black
    
    prompts = [("chess piece", 1)]
    
    for prompt_text, color_id in prompts:
        processor = Sam3Processor(image_model)
        inference_state = processor.set_image(pil_img)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
        
        boxes = output["boxes"]
        scores = output["scores"]
        
        high_conf_indices = scores > 0.4
        
        if high_conf_indices.sum() > 0:
            all_boxes.append(boxes[high_conf_indices])
            all_scores.append(scores[high_conf_indices])
            all_colors.extend([color_id] * len(boxes[high_conf_indices]))
    
    if len(all_boxes) > 0:
        all_boxes = torch.cat(all_boxes, dim=0) if isinstance(all_boxes[0], torch.Tensor) else np.vstack(all_boxes)
        
        if isinstance(all_boxes, torch.Tensor):
            all_boxes = all_boxes.cpu().numpy()
        
        for i, (box, color_id) in enumerate(zip(all_boxes, all_colors)):
            x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
            
            feet_x = (x1 + x2) / 2
            feet_y = y2 
            
            col = int(feet_x // square_size)
            row = int(feet_y // square_size)
            
            col = np.clip(col, 0, 7)
            row = np.clip(row, 0, 7)
            
            board_state[row, col] = color_id
            
            # Visualize Piece Bboxes and Feet
            color = (0, 255, 0) if color_id == 1 else (255, 0, 0)
            cv2.circle(vis_img, (int(feet_x), int(feet_y)), 5, color, -1)
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    return board_state, vis_img

# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------

# Configuration
device = "cuda"
dtype = torch.bfloat16
MAX_FRAMES = None
DETECTION_PROMPT = "chessboard"
WARP_SIZE = 640

# Load models
print("Loading SAM3 image model for detection...")
image_model = build_sam3_image_model()
# Don't create processor here - we'll create fresh ones as needed

print("Loading SAM3 tracker model for video tracking...")
tracker_model = Sam3TrackerVideoModel.from_pretrained(
    "facebook/sam3",
    torch_dtype=dtype,
).to(device).eval()
tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

# Open video
cap = cv2.VideoCapture("test1.mp4")
ret, first_frame = cap.read()
assert ret, "Failed to read first frame"

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (width, height))

# ---------------------------------------------
# DETECT CHESSBOARD (Frame 0)
# ---------------------------------------------
print(f"Detecting chessboard with prompt: '{DETECTION_PROMPT}'")

# Create a processor for initial board detection
initial_processor = Sam3Processor(image_model)

first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(first_frame_rgb)

inference_state = initial_processor.set_image(pil_image)
output = initial_processor.set_text_prompt(state=inference_state, prompt=DETECTION_PROMPT)

masks = output["masks"]
scores = output["scores"]

if len(masks) == 0:
    raise ValueError(f"No chessboard detected with prompt '{DETECTION_PROMPT}'.")

# Best detection
best_idx = scores.argmax()
best_mask = masks[best_idx]

# Process mask
if isinstance(best_mask, torch.Tensor):
    mask_np = best_mask.cpu().numpy()
else:
    mask_np = best_mask

if mask_np.ndim > 2: 
    mask_np = mask_np.squeeze()
if mask_np.shape != (first_frame.shape[0], first_frame.shape[1]):
    mask_np = cv2.resize(mask_np, (first_frame.shape[1], first_frame.shape[0]))

if mask_np.max() <= 1.0: 
    mask_np = (mask_np * 255).astype(np.uint8)
else: 
    mask_np = mask_np.astype(np.uint8)

# Sample points for tracking
y_coords, x_coords = np.where(mask_np > 0)
indices = np.linspace(0, len(x_coords) - 1, 16, dtype=int)
points = [[int(x_coords[i]), int(y_coords[i])] for i in indices]

# ---------------------------------------------
# INIT TRACKER
# ---------------------------------------------
print("Initializing video tracking session...")
session = tracker_processor.init_video_session(inference_device=device, dtype=dtype)

inputs0 = tracker_processor(images=first_frame, return_tensors="pt").to(device, dtype=dtype)

tracker_processor.add_inputs_to_inference_session(
    inference_session=session,
    frame_idx=0,
    obj_ids=1,
    input_points=[[[pt for pt in points]]],
    input_labels=[[[1] * len(points)]],
    original_size=inputs0.original_sizes[0],
)

# Segment Frame 0
with torch.no_grad():
    _ = tracker_model(inference_session=session, frame=inputs0.pixel_values[0])






# ---------------------------------------------
# TRACKING LOOP (Modified for Grid Overlay)
# ---------------------------------------------
frame_idx = 0

while True:
    if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
        break
    
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_idx += 1
    inputs = tracker_processor(images=frame, return_tensors="pt").to(device, dtype=dtype)
    
    with torch.no_grad():
        output = tracker_model(inference_session=session, frame=inputs.pixel_values[0])

    masks_out = tracker_processor.post_process_masks(
        [output.pred_masks], 
        original_sizes=[[frame.shape[0], frame.shape[1]]], 
        binarize=True
    )[0]
    
    current_mask = masks_out[0, 0].cpu().numpy().astype(np.uint8) * 255
    overlay = frame.copy()
    corners = get_board_corners(current_mask)
    
    if corners is not None:
        # Warp with 15% margin to get the Matrix
        warped_board, Matrix = get_birdseye_view(frame, corners, size=WARP_SIZE, margin_ratio=0.15)
        
        # --- NEW: Calculate Grid in Original Frame Space ---
        inv_M = cv2.invert(Matrix)[1]
        margin = int(WARP_SIZE * 0.15)
        board_span = WARP_SIZE - (2 * margin)
        square_size = board_span / 8

        for i in range(9):
            # Define line endpoints in warped space
            v_start = np.array([[[margin + i * square_size, margin]]], dtype="float32")
            v_end   = np.array([[[margin + i * square_size, WARP_SIZE - margin]]], dtype="float32")
            h_start = np.array([[[margin, margin + i * square_size]]], dtype="float32")
            h_end   = np.array([[[WARP_SIZE - margin, margin + i * square_size]]], dtype="float32")

            # Project endpoints back to original frame
            v_s = cv2.perspectiveTransform(v_start, inv_M)[0][0].astype(int)
            v_e = cv2.perspectiveTransform(v_end, inv_M)[0][0].astype(int)
            h_s = cv2.perspectiveTransform(h_start, inv_M)[0][0].astype(int)
            h_e = cv2.perspectiveTransform(h_end, inv_M)[0][0].astype(int)

            # Draw onto the overlay
            cv2.line(overlay, tuple(v_s), tuple(v_e), (0, 255, 0), 2)
            cv2.line(overlay, tuple(h_s), tuple(h_e), (0, 255, 0), 2)
        # ---------------------------------------------------

        # Detect pieces and save debug images
        if frame_idx % 30 == 0:
            grid_state, vis_img = update_board_state_sam3(warped_board, image_model, board_size=WARP_SIZE)
            cv2.imwrite(f"frame_debug_{frame_idx}.jpg", overlay) # Now includes the projected grid
            cv2.imwrite(f"state_debug_{frame_idx}.jpg", vis_img)
            
        thumb = cv2.resize(warped_board, (200, 200))
        overlay[0:200, 0:200] = thumb
    
    out.write(overlay)
    print(f"Processed frame {frame_idx}")

cap.release()
out.release()








# # ---------------------------------------------
# # TRACKING LOOP
# # ---------------------------------------------
# frame_idx = 0

# while True:
#     if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
#         break
    
#     ret, frame = cap.read()
#     if not ret:
#         break
        
#     frame_idx += 1

#     # TRACK
#     inputs = tracker_processor(images=frame, return_tensors="pt").to(device, dtype=dtype)
    
#     with torch.no_grad():
#         output = tracker_model(inference_session=session, frame=inputs.pixel_values[0])

#     # POST-PROCESS MASK
#     masks_out = tracker_processor.post_process_masks(
#         [output.pred_masks], 
#         original_sizes=[[frame.shape[0], frame.shape[1]]], 
#         binarize=True
#     )[0]
    
#     current_mask = masks_out[0, 0].cpu().numpy().astype(np.uint8) * 255
    
#     # ---------------------------------------------
#     # GEOMETRIC RECTIFICATION
#     # ---------------------------------------------
#     overlay = frame.copy()
#     corners = get_board_corners(current_mask)
    
#     if corners is None:
#         print(f"Frame {frame_idx}: Could not find board corners.")    
#     else:
#         # Visualize detection
#         cv2.polylines(overlay, [corners], True, (0, 255, 255), 3) 
        
#         # Warp with 15% margin
#         warped_board, Matrix = get_birdseye_view(frame, corners, size=WARP_SIZE, margin_ratio=0.15)
        
#         # Draw grid
#         debug_view = draw_grid_on_warped(warped_board, size=WARP_SIZE, margin_ratio=0.15)
        
#         # Detect pieces every 30 frames
#         if frame_idx % 30 == 0:
#             print(f"Frame {frame_idx}: Scanning for pieces...")
            
#             # KEY FIX: Pass image_model instead of processor
#             grid_state, vis_img = update_board_state_sam3(warped_board, image_model, board_size=WARP_SIZE)
            
#             print("\nCurrent Board State (1=Occupied):")
#             print(grid_state)
            
#             cv2.imwrite(f"state_debug_{frame_idx}.jpg", vis_img)
            
#         # Picture-in-Picture overlay
#         thumb = cv2.resize(warped_board, (200, 200))
#         overlay[0:200, 0:200] = thumb
    
#     # Write to video
#     out.write(overlay)
#     print(f"Processed frame {frame_idx}")

# cap.release()
# out.release()
# print("Done!")