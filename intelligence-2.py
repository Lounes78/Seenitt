import torch
import cv2
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# ---------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------
def is_board_fully_visible(mask, frame_w, frame_h, margin=10):
    """
    Checks if the chessboard mask is fully inside the frame with a safety margin.
    Returns True if visible and safe, False if touching edges.
    """
    if np.count_nonzero(mask) == 0:
        return False
        
    # Get Bounding Box from Mask
    x, y, w, h = cv2.boundingRect(mask)
    x2 = x + w
    y2 = y + h
    
    # Check distances to all 4 margins
    # Left > margin, Top > margin, Right < Width - margin, Bottom < Height - margin
    if (x > margin) and (y > margin) and (x2 < frame_w - margin) and (y2 < frame_h - margin):
        return True
    return False

def order_points(pts):
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
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    perimeter = cv2.arcLength(hull, True)
    for eps_factor in np.linspace(0.02, 0.1, 10):
        epsilon = eps_factor * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return np.int32(box)

def get_birdseye_view(img, corners, size=640, margin_ratio=0.15):
    rect = order_points(corners)
    margin = int(size * margin_ratio)
    dst = np.array([
        [margin, margin],
        [size - margin - 1, margin],
        [size - margin - 1, size - margin - 1],
        [margin, size - margin - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (size, size)), M

def update_board_state_sam3(warped_img, image_model):
    img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    vis_img = warped_img.copy()
    h, w = warped_img.shape[:2]
    board_state = np.zeros((8, 8), dtype=int)
    
    prompts = [("chess piece", 1)]
    all_boxes, all_scores, all_colors = [], [], []
    
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
        if isinstance(all_boxes, torch.Tensor): all_boxes = all_boxes.cpu().numpy()
        
        for i, (box, color_id) in enumerate(zip(all_boxes, all_colors)):
            x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
            feet_x, feet_y = (x1 + x2) / 2, y2 
            
            col = int((feet_x / w) * 8)
            row = int((feet_y / h) * 8)
            board_state[np.clip(row, 0, 7), np.clip(col, 0, 7)] = color_id
            
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(vis_img, (int(feet_x), int(feet_y)), 5, (0,0,255), -1)
            cv2.putText(vis_img, f"({row},{col})", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    
    return board_state, vis_img

# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------
device = "cuda"
dtype = torch.bfloat16
MAX_FRAMES = None
DETECTION_PROMPT = "chessboard"
WARP_SIZE = 640
FRAME_MARGIN_THRESHOLD = 15 # Pixels required between board bbox and frame edge

print("Loading Models...")
image_model = build_sam3_image_model()
tracker_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device).eval()
tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

cap = cv2.VideoCapture("test1.mp4")
ret, first_frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_tracked.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# --- INITIAL DETECTION ---
print("Initial Detection...")
init_proc = Sam3Processor(image_model)
inf_state = init_proc.set_image(Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)))
out_init = init_proc.set_text_prompt(state=inf_state, prompt=DETECTION_PROMPT)

best_mask = out_init["masks"][out_init["scores"].argmax()]
if isinstance(best_mask, torch.Tensor): best_mask = best_mask.cpu().numpy()
if best_mask.ndim > 2: best_mask = best_mask.squeeze()
mask_np = (best_mask * 255).astype(np.uint8) if best_mask.max() <= 1.0 else best_mask.astype(np.uint8)
mask_np = cv2.resize(mask_np, (width, height))

y_c, x_c = np.where(mask_np > 0)
points = [[int(x_c[i]), int(y_c[i])] for i in np.linspace(0, len(x_c)-1, 16, dtype=int)]

# --- INIT TRACKER ---
session = tracker_processor.init_video_session(inference_device=device, dtype=dtype)
inputs0 = tracker_processor(images=first_frame, return_tensors="pt").to(device, dtype=dtype)
tracker_processor.add_inputs_to_inference_session(
    inference_session=session, frame_idx=0, obj_ids=1,
    input_points=[[points]], input_labels=[[[1]*len(points)]], original_size=inputs0.original_sizes[0]
)
with torch.no_grad(): tracker_model(inference_session=session, frame=inputs0.pixel_values[0])

# --- LOOP ---
frame_idx = 0
while True:
    if MAX_FRAMES and frame_idx >= MAX_FRAMES: break
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    # Track
    inputs = tracker_processor(images=frame, return_tensors="pt").to(device, dtype=dtype)
    with torch.no_grad(): output = tracker_model(inference_session=session, frame=inputs.pixel_values[0])
    
    masks_out = tracker_processor.post_process_masks([output.pred_masks], original_sizes=[[height, width]], binarize=True)[0]
    current_mask = masks_out[0, 0].cpu().numpy().astype(np.uint8) * 255

    overlay = frame.copy()
    corners = get_board_corners(current_mask)
    
    if corners is not None:
        cv2.polylines(overlay, [corners], True, (0, 255, 255), 3)
        warped_board, _ = get_birdseye_view(frame, corners, size=WARP_SIZE)
        
        # --- NEW VISIBILITY CHECK ---
        fully_visible = is_board_fully_visible(current_mask, width, height, margin=FRAME_MARGIN_THRESHOLD)

        if frame_idx % 1 == 0:
            if fully_visible:
                print(f"Frame {frame_idx}: Board fully visible. Detecting pieces...")
                grid_state, vis_img = update_board_state_sam3(warped_board, image_model)
                
                # --- PRINT THE BOARD STATE ---
                print(f"\n--- Board State Frame {frame_idx} ---")
                print(grid_state)
                print("---------------------------------")
                
                cv2.imwrite(f"state_debug_{frame_idx}.jpg", vis_img)
            else:
                print(f"Frame {frame_idx}: Skipped detection (Board partially out of frame).")
                cv2.putText(overlay, "BOARD CLIPPED - SKIPPING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        thumb = cv2.resize(warped_board, (200, 200))
        overlay[0:200, 0:200] = thumb

    out.write(overlay)
    print(f"Processed frame {frame_idx}")

cap.release()
out.release()
print("Done!")