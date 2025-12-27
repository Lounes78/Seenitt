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
    if np.count_nonzero(mask) == 0:
        return False
    x, y, w, h = cv2.boundingRect(mask)
    x2 = x + w
    y2 = y + h
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

def is_point_in_polygon(point, polygon):
    """Check if a point (x, y) is inside a polygon defined by 'polygon'."""
    if polygon is None:
        return False
    pt_tuple = (float(point[0]), float(point[1]))
    return cv2.pointPolygonTest(polygon, pt_tuple, False) >= 0

def find_extreme_corners_with_boxes(feet_points, boxes, board_corners):
    """
    1. Filters input points to keep ONLY those inside the board polygon.
    2. Finds the 4 extreme pieces (TL, TR, BL, BR) based on feet position.
    3. Returns specific bbox points for those pieces based on user rules.
    """
    # --- STEP 1: Filter points (and keep corresponding boxes) ---
    valid_data = [] # Stores (foot_point, box)
    for pt, box in zip(feet_points, boxes):
        if is_point_in_polygon(pt, board_corners):
            valid_data.append((pt, box))
            
    if len(valid_data) < 4:
        return None

    # Separate back into lists for easier indexing
    valid_feet = [d[0] for d in valid_data]
    valid_boxes = [d[1] for d in valid_data]
    
    feet_np = np.array(valid_feet)

    # --- STEP 2: Find indices of extreme pieces ---
    tl_idx = np.argmin(feet_np[:, 0] + feet_np[:, 1]) # TL: Min(x + y)
    br_idx = np.argmax(feet_np[:, 0] + feet_np[:, 1]) # BR: Max(x + y)
    tr_idx = np.argmax(feet_np[:, 0] - feet_np[:, 1]) # TR: Max(x - y)
    bl_idx = np.argmin(feet_np[:, 0] - feet_np[:, 1]) # BL: Min(x - y)
    
    # Get the bounding boxes for these extreme pieces
    box_tl = valid_boxes[tl_idx]
    box_tr = valid_boxes[tr_idx]
    box_bl = valid_boxes[bl_idx]
    box_br = valid_boxes[br_idx]
    
    # --- STEP 3: Calculate Grid Corners based on BBox Rules ---
    # Box format is [x1, y1, x2, y2]
    
    h_tl = box_tl[3] - box_tl[1]
    pt_tl = np.array([box_tl[0], box_tl[1] + 0.7 * h_tl])

    h_tr = box_tr[3] - box_tr[1]
    pt_tr = np.array([box_tr[2], box_tr[1] + 0.7 * h_tr])

    pt_bl = np.array([box_bl[0], box_bl[3]])
    pt_br = np.array([box_br[2], box_br[3]])
    
    return pt_tl, pt_tr, pt_bl, pt_br

def interpolate_grid_points(p1, p2, steps=8):
    return np.linspace(p1, p2, steps + 1)

def update_board_state_sam3(warped_img, image_model, warped_mask):
    img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    vis_img = warped_img.copy()
    h, w = warped_img.shape[:2]
    board_state = np.zeros((8, 8), dtype=int)

    # --- FILL HOLES ---
    cnts, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest_cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(warped_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)

    # --- CALCULATE BOARD CORNERS ---
    board_corners = get_board_corners(warped_mask)

    # 1. Detect Pieces
    prompts = [("chess piece", 1)]
    all_boxes, all_scores, all_colors = [], [], []
    
    processor = Sam3Processor(image_model)
    inference_state = processor.set_image(pil_img)
    
    for prompt_text, color_id in prompts:
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
        boxes = output["boxes"]
        scores = output["scores"]
        high_conf_indices = scores > 0.4
        
        if high_conf_indices.sum() > 0:
            all_boxes.append(boxes[high_conf_indices])
            all_scores.append(scores[high_conf_indices])
            all_colors.extend([color_id] * len(boxes[high_conf_indices]))

    if len(all_boxes) == 0:
        return board_state, vis_img

    all_boxes = torch.cat(all_boxes, dim=0) if isinstance(all_boxes[0], torch.Tensor) else np.vstack(all_boxes)
    if isinstance(all_boxes, torch.Tensor): all_boxes = all_boxes.cpu().numpy()

    feet_list = []
    box_list = []
    for box in all_boxes:
        x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
        # ADJUSTED FEET: Moves the point slightly up from bottom to reduce perspective error
        feet_x = (x1 + x2) / 2
        feet_y = y1 + (y2 - y1) * 0.9  # 90% down the box instead of 100%
        feet_list.append([feet_x, feet_y])
        box_list.append([x1, y1, x2, y2])

    # 2. Determine Grid using NEW logic
    corner_result = find_extreme_corners_with_boxes(feet_list, box_list, board_corners)
    
    if corner_result is not None:
        pt_tl, pt_tr, pt_bl, pt_br = corner_result
        
        # Draw Grid Lines
        top_edge = interpolate_grid_points(pt_tl, pt_tr)
        bottom_edge = interpolate_grid_points(pt_bl, pt_br)
        left_edge = interpolate_grid_points(pt_tl, pt_bl)
        right_edge = interpolate_grid_points(pt_tr, pt_br)

        for i in range(9):
            cv2.line(vis_img, tuple(top_edge[i].astype(int)), tuple(bottom_edge[i].astype(int)), (255, 0, 0), 2)
            cv2.line(vis_img, tuple(left_edge[i].astype(int)), tuple(right_edge[i].astype(int)), (255, 0, 0), 2)
            
        for pt in [pt_tl, pt_tr, pt_bl, pt_br]:
             cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), -1)

        # Map pieces using POLYGON LOGIC (Grid Lines)
        for i, (box, feet, color_id) in enumerate(zip(all_boxes, feet_list, all_colors)):
            fx, fy = feet
            
            # --- DETERMINE COLUMN ---
            col = -1
            for c in range(8):
                # Polygon for this column strip: Top[c] -> Top[c+1] -> Bot[c+1] -> Bot[c]
                p1 = top_edge[c]
                p2 = top_edge[c+1]
                p3 = bottom_edge[c+1]
                p4 = bottom_edge[c]
                poly_col = np.array([p1, p2, p3, p4], dtype=np.int32)
                if cv2.pointPolygonTest(poly_col.astype(np.float32), (float(fx), float(fy)), False) >= 0:
                    col = c
                    break
            
            # Fallback for clipping (nearest side)
            if col == -1:
                col = 0 if fx < (pt_tl[0] + pt_tr[0]) / 2 else 7

            # --- DETERMINE ROW ---
            row = -1
            for r in range(8):
                # Polygon for this row strip: Left[r] -> Right[r] -> Right[r+1] -> Left[r+1]
                p1 = left_edge[r]
                p2 = right_edge[r]
                p3 = right_edge[r+1]
                p4 = left_edge[r+1]
                poly_row = np.array([p1, p2, p3, p4], dtype=np.int32)
                if cv2.pointPolygonTest(poly_row.astype(np.float32), (float(fx), float(fy)), False) >= 0:
                    row = r
                    break
            
            # Fallback for clipping
            if row == -1:
                row = 0 if fy < (pt_tl[1] + pt_bl[1]) / 2 else 7

            # Update State
            board_state[row, col] = color_id
            
            x1, y1, x2, y2 = box
            # cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            
            if is_point_in_polygon(feet, board_corners):
                cv2.circle(vis_img, (int(fx), int(fy)), 5, (0,255,0), -1) 
            else:
                cv2.circle(vis_img, (int(fx), int(fy)), 5, (0,0,255), -1) 
                
            cv2.putText(vis_img, f"{row},{col}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    else:
        print("Warning: Could not find 4 valid corners inside mask.")
        cv2.putText(vis_img, "NO VALID CORNERS IN MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for box, feet in zip(all_boxes, feet_list):
            # cv2.rectangle(vis_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.circle(vis_img, (int(feet[0]), int(feet[1])), 5, (0,0,255), -1)

    return board_state, vis_img

# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------
device = "cuda"
dtype = torch.bfloat16
MAX_FRAMES = None
DETECTION_PROMPT = "chessboard"
WARP_SIZE = 640
FRAME_MARGIN_THRESHOLD = 15

print("Loading Models...")
image_model = build_sam3_image_model()
tracker_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device).eval()
tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

cap = cv2.VideoCapture("test2.mp4")
ret, first_frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

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

    inputs = tracker_processor(images=frame, return_tensors="pt").to(device, dtype=dtype)
    with torch.no_grad(): output = tracker_model(inference_session=session, frame=inputs.pixel_values[0])
    
    masks_out = tracker_processor.post_process_masks([output.pred_masks], original_sizes=[[height, width]], binarize=True)[0]
    current_mask = masks_out[0, 0].cpu().numpy().astype(np.uint8) * 255

    overlay = frame.copy()
    
    # Mask visualization and contour drawing removed here

    corners = get_board_corners(current_mask)
    
    if corners is not None:
        cv2.polylines(overlay, [corners], True, (0, 255, 255), 3)
        warped_board, M = get_birdseye_view(frame, corners, size=WARP_SIZE)
        warped_mask = cv2.warpPerspective(current_mask, M, (WARP_SIZE, WARP_SIZE))
        
        fully_visible = is_board_fully_visible(current_mask, width, height, margin=FRAME_MARGIN_THRESHOLD)

        if frame_idx % 1 == 0:
            if fully_visible:
                print(f"Frame {frame_idx}: Board fully visible. Detecting pieces & building grid...")
                grid_state, vis_img = update_board_state_sam3(warped_board, image_model, warped_mask)
                print(f"--- Frame {frame_idx} Done ---")
                print(grid_state)
                print("---------------------------------")                
                cv2.imwrite(f"state_debug_{frame_idx}.jpg", vis_img)

            else:
                print(f"Frame {frame_idx}: Skipped (Clipped).")
                cv2.putText(overlay, "BOARD CLIPPED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if 'warped_board' in locals():
            thumb = cv2.resize(warped_board, (200, 200))
            overlay[0:200, 0:200] = thumb

    out.write(overlay)
    print(f"Processed frame {frame_idx}")

cap.release()
out.release()
print("Done!")