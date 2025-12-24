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
    
    KEY FIX: Process each prompt separately and create fresh processor
    """
    # Prepare image
    img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Initialize empty 8x8 board
    board_state = np.zeros((8, 8), dtype=int)
    square_size = board_size / 8
    vis_img = warped_img.copy()
    
    # Collect all detections
    all_boxes = []
    all_scores = []
    all_colors = []  # 1 for white, 2 for black
    
    # Process each color separately with its own processor instance
    prompts = [
        # ("chess piece", 1),  
        ("white chess piece", 1),
        ("black chess piece", 2),
    ]
    
    for prompt_text, color_id in prompts:
        # Create fresh processor for each prompt
        processor = Sam3Processor(image_model)
        
        # Set image and prompt
        inference_state = processor.set_image(pil_img)
        
        # Use single string prompt instead of list
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
        
        boxes = output["boxes"]
        scores = output["scores"]
        
        # Filter by confidence
        high_conf_indices = scores > 0.4
        
        if high_conf_indices.sum() > 0:
            filtered_boxes = boxes[high_conf_indices]
            filtered_scores = scores[high_conf_indices]
            
            # Add to collections
            all_boxes.append(filtered_boxes)
            all_scores.append(filtered_scores)
            all_colors.extend([color_id] * len(filtered_boxes))
    
    # Combine all detections
    if len(all_boxes) > 0:
        all_boxes = torch.cat(all_boxes, dim=0) if isinstance(all_boxes[0], torch.Tensor) else np.vstack(all_boxes)
        all_scores = torch.cat(all_scores, dim=0) if isinstance(all_scores[0], torch.Tensor) else np.concatenate(all_scores)
        
        # Convert to numpy if needed
        if isinstance(all_boxes, torch.Tensor):
            all_boxes = all_boxes.cpu().numpy()
        if isinstance(all_scores, torch.Tensor):
            all_scores = all_scores.cpu().numpy()
        
        # Process each detection
        for i, (box, color_id) in enumerate(zip(all_boxes, all_colors)):
            x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
            
            # Find the piece's feet (bottom center of bounding box)
            feet_x = (x1 + x2) / 2
            feet_y = y2  # Bottom-most point
            
            # Map to grid
            col = int(feet_x // square_size)
            row = int(feet_y // square_size)
            
            # Safety check for boundaries
            col = np.clip(col, 0, 7)
            row = np.clip(row, 0, 7)
            
            # Mark the board
            board_state[row, col] = color_id
            
            # Visualize
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
# TRACKING LOOP
# ---------------------------------------------
frame_idx = 0

while True:
    if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
        break
    
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_idx += 1

    # TRACK
    inputs = tracker_processor(images=frame, return_tensors="pt").to(device, dtype=dtype)
    
    with torch.no_grad():
        output = tracker_model(inference_session=session, frame=inputs.pixel_values[0])

    # POST-PROCESS MASK
    masks_out = tracker_processor.post_process_masks(
        [output.pred_masks], 
        original_sizes=[[frame.shape[0], frame.shape[1]]], 
        binarize=True
    )[0]
    
    current_mask = masks_out[0, 0].cpu().numpy().astype(np.uint8) * 255
    
    # ---------------------------------------------
    # GEOMETRIC RECTIFICATION
    # ---------------------------------------------
    overlay = frame.copy()
    corners = get_board_corners(current_mask)
    
    if corners is None:
        print(f"Frame {frame_idx}: Could not find board corners.")    
    else:
        # Visualize detection
        cv2.polylines(overlay, [corners], True, (0, 255, 255), 3) 
        
        # Warp with 15% margin
        warped_board, Matrix = get_birdseye_view(frame, corners, size=WARP_SIZE, margin_ratio=0.15)
        
        # Draw grid
        debug_view = draw_grid_on_warped(warped_board, size=WARP_SIZE, margin_ratio=0.15)
        
        # Detect pieces every 30 frames
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: Scanning for pieces...")
            
            # KEY FIX: Pass image_model instead of processor
            grid_state, vis_img = update_board_state_sam3(warped_board, image_model, board_size=WARP_SIZE)
            
            print("\nCurrent Board State (1=Occupied):")
            print(grid_state)
            
            cv2.imwrite(f"state_debug_{frame_idx}.jpg", vis_img)
            
        # Picture-in-Picture overlay
        thumb = cv2.resize(warped_board, (200, 200))
        overlay[0:200, 0:200] = thumb
    
    # Write to video
    out.write(overlay)
    print(f"Processed frame {frame_idx}")

cap.release()
out.release()
print("Done!")






# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor
# from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# # ---------------------------------------------
# # HELPER FUNCTIONS FOR GEOMETRIC RECTIFICATION
# # ---------------------------------------------
# def order_points(pts):
#     """Sorts points: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]      # TL
#     rect[2] = pts[np.argmax(s)]      # BR
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]   # TR
#     rect[3] = pts[np.argmax(diff)]   # BL
#     return rect


# def get_board_corners(mask_binary):
#     contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours: 
#         return None
    
#     # Get the largest contour (the board)
#     largest = max(contours, key=cv2.contourArea)
    
#     # 1. Smooth the shape using Convex Hull
#     hull = cv2.convexHull(largest)
#     perimeter = cv2.arcLength(hull, True)
    
#     # 2. Adaptive approximation: Try to find 4 corners
#     for eps_factor in np.linspace(0.02, 0.1, 10):
#         epsilon = eps_factor * perimeter
#         approx = cv2.approxPolyDP(hull, epsilon, True)
        
#         if len(approx) == 4:
#             return approx.reshape(4, 2)
            
#     # 3. Fallback: Rotated Bounding Box
#     rect = cv2.minAreaRect(largest)
#     box = cv2.boxPoints(rect)
#     # FIX: Use int32 or astype(int) instead of np.int0
#     box = np.int32(box) 
#     return box

# def get_birdseye_view(img, corners, size=640, margin_ratio=0.15):
#     """
#     Warps the image to a top-down view with padding to keep piece heads visible.
#     margin_ratio: Percentage of image to leave as margin (0.15 = 15%)
#     """
#     rect = order_points(corners)
    
#     # Calculate margin in pixels
#     margin = int(size * margin_ratio)
    
#     # Define destination points shifted INWARD by the margin
#     # This forces the warp to include pixels 'outside' the detected corners
#     dst = np.array([
#         [margin, margin],                # TL
#         [size - margin - 1, margin],     # TR
#         [size - margin - 1, size - margin - 1], # BR
#         [margin, size - margin - 1]      # BL
#     ], dtype="float32")
    
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(img, M, (size, size))
#     return warped, M

# def draw_grid_on_warped(warped_img, size=640, margin_ratio=0.15):
#     """Draws the 8x8 grid on the warped image to verify alignment."""
#     vis = warped_img.copy()
#     margin = int(size * margin_ratio)
#     board_span = size - (2 * margin)
#     square_size = board_span / 8
    
#     # Draw vertical lines
#     for i in range(9):
#         x = int(margin + i * square_size)
#         cv2.line(vis, (x, margin), (x, size - margin), (0, 255, 0), 2)
        
#     # Draw horizontal lines
#     for i in range(9):
#         y = int(margin + i * square_size)
#         cv2.line(vis, (margin, y), (size - margin, y), (0, 255, 0), 2)
        
#     return vis




# def update_board_state_sam3(warped_img, image_processor, board_size=640):
#     """
#     Detects pieces using SAM3, finds their 'feet', and maps them to the 8x8 grid.
#     Returns: 8x8 Numpy array (0=Empty, 1=White, 2=Black)
#     """
#     # 1. Prepare image for SAM3
#     img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(img_rgb)
    
#     # 2. Prompt SAM3 to find pieces
#     # Note: We use specific prompts to distinguish colors
#     inference_state = image_processor.set_image(pil_img)
    
#     # We ask for white and black pieces separately to get their color class
#     prompts = ["white chess piece", "black chess piece"]
#     output = image_processor.set_text_prompt(state=inference_state, prompt=prompts)
    
#     boxes = output["boxes"]
#     scores = output["scores"]
#     class_ids = output["label_ids"] # Assuming prompt index corresponds to class ID
    
#     # Initialize empty 8x8 board
#     board_state = np.zeros((8, 8), dtype=int)
#     square_size = board_size / 8
    
#     # 3. Process detections
#     # Threshold to filter weak detections
#     high_conf_indices = scores > 0.4
    
#     filtered_boxes = boxes[high_conf_indices]
#     # Note: You might need to map scores/indices back to which prompt (white/black) was used.
#     # SAM3 output structure can vary; usually it returns [N, 4] boxes.
#     # For simplicity here, let's assume we handle all detections generic first or check the return structure.
    
#     # If the model returns grouped masks per prompt, we might need to iterate differently.
#     # Let's assume for this snippet we just get 'boxes' and we want to Map them.
    
#     for i, box in enumerate(filtered_boxes):
#         x1, y1, x2, y2 = box.tolist()
        
#         # 4. FIND THE FEET (Robust Logic)
#         # The piece is located at the bottom center of its bounding box
#         feet_x = (x1 + x2) / 2
#         feet_y = y2  # The bottom-most point
        
#         # 5. MAP TO GRID
#         col = int(feet_x // square_size)
#         row = int(feet_y // square_size)
        
#         # Safety check for boundaries
#         col = np.clip(col, 0, 7)
#         row = np.clip(row, 0, 7)
        
#         # Mark the board
#         # (For now we mark 1 for occupied, later we can distinguish W/B)
#         board_state[row, col] = 1 
        
#         # visualize feet
#         cv2.circle(warped_img, (int(feet_x), int(feet_y)), 5, (0, 0, 255), -1)
        
#     return board_state, warped_img

# # ---------------------------------------------
# # MAIN PIPELINE
# # ---------------------------------------------

# # Configuration
# device = "cuda"
# dtype = torch.bfloat16
# MAX_FRAMES = 500
# DETECTION_PROMPT = "chessboard"
# WARP_SIZE = 640 # Size of the rectified board image

# # Load models
# print("Loading SAM3 image model for detection...")
# image_model = build_sam3_image_model()
# image_processor = Sam3Processor(image_model)

# print("Loading SAM3 tracker model for video tracking...")
# tracker_model = Sam3TrackerVideoModel.from_pretrained(
#     "facebook/sam3",
#     torch_dtype=dtype,
# ).to(device).eval()
# tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

# # Open video
# cap = cv2.VideoCapture("test1.mp4")
# ret, first_frame = cap.read()
# assert ret, "Failed to read first frame"

# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Setup video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (width, height))

# # ---------------------------------------------
# # DETECT CHESSBOARD (Frame 0)
# # ---------------------------------------------
# print(f"Detecting chessboard with prompt: '{DETECTION_PROMPT}'")
# first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
# pil_image = Image.fromarray(first_frame_rgb)

# inference_state = image_processor.set_image(pil_image)
# output = image_processor.set_text_prompt(state=inference_state, prompt=DETECTION_PROMPT)

# masks = output["masks"]
# scores = output["scores"]

# if len(masks) == 0:
#     raise ValueError(f"No chessboard detected with prompt '{DETECTION_PROMPT}'.")

# # Best detection
# best_idx = scores.argmax()
# best_mask = masks[best_idx]

# # Process mask
# if isinstance(best_mask, torch.Tensor):
#     mask_np = best_mask.cpu().numpy()
# else:
#     mask_np = best_mask

# if mask_np.ndim > 2: mask_np = mask_np.squeeze()
# if mask_np.shape != (first_frame.shape[0], first_frame.shape[1]):
#     mask_np = cv2.resize(mask_np, (first_frame.shape[1], first_frame.shape[0]))

# if mask_np.max() <= 1.0: mask_np = (mask_np * 255).astype(np.uint8)
# else: mask_np = mask_np.astype(np.uint8)

# # Sample points for tracking
# y_coords, x_coords = np.where(mask_np > 0)
# indices = np.linspace(0, len(x_coords) - 1, 16, dtype=int)
# points = [[int(x_coords[i]), int(y_coords[i])] for i in indices]

# # ---------------------------------------------
# # INIT TRACKER
# # ---------------------------------------------
# print("Initializing video tracking session...")
# session = tracker_processor.init_video_session(inference_device=device, dtype=dtype)

# inputs0 = tracker_processor(images=first_frame, return_tensors="pt").to(device, dtype=dtype)

# tracker_processor.add_inputs_to_inference_session(
#     inference_session=session,
#     frame_idx=0,
#     obj_ids=1,
#     input_points=[[[pt for pt in points]]],
#     input_labels=[[[1] * len(points)]],
#     original_size=inputs0.original_sizes[0],
# )

# # Segment Frame 0 to prime the memory
# with torch.no_grad():
#     _ = tracker_model(inference_session=session, frame=inputs0.pixel_values[0])

# # ---------------------------------------------
# # TRACKING LOOP
# # ---------------------------------------------
# frame_idx = 0 
# # Note: Re-reading from frame 0 or continuing depends on use case. 
# # Usually we track from the NEXT frame, but let's just reset loop variable for clarity.
# # Since we already read frame 0, we can start loop from 0 or 1.
# # Here we continue reading from stream.

# while True:
#     if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
#         break
    
#     # We already processed frame 0 logic above, but for the loop 
#     # we need to read the NEXT frame from 'cap'.
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
#         # 1. Visualize detection on original frame
#         cv2.polylines(overlay, [corners], True, (0, 255, 255), 3) 
        
#         # 2. Warp with 15% margin to keep piece heads
#         warped_board, Matrix = get_birdseye_view(frame, corners, size=WARP_SIZE, margin_ratio=0.15)
        
#         # 3. Draw grid to verify fit
#         debug_view = draw_grid_on_warped(warped_board, size=WARP_SIZE, margin_ratio=0.15)
        
#         if frame_idx % 30 == 0:
#             print(f"Frame {frame_idx}: Scanning for pieces...")
            
#             # Detect pieces and map to grid
#             # Note: We pass 'image_processor' which is the SAM3 Image model, not the tracker
#             grid_state, vis_img = update_board_state_sam3(warped_board, image_processor, board_size=WARP_SIZE)
            
#             print("\nCurrent Board State (1=Occupied):")
#             print(grid_state)
            
#             cv2.imwrite(f"state_debug_{frame_idx}.jpg", vis_img)
            
                            
#         # Optional: Picture-in-Picture (Overlay small warped board on video)
#         # Resize warped to small thumbnail
#         thumb = cv2.resize(warped_board, (200, 200))
#         overlay[0:200, 0:200] = thumb
    
#     # Write to video
#     out.write(overlay)
#     print(f"Processed frame {frame_idx}")

# cap.release()
# out.release()
# print("Done.")