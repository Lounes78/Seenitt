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

def get_solid_mask(mask_binary):
    """Fills holes in the mask using Convex Hull for robust corner/feet checking."""
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return mask_binary
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    solid_mask = np.zeros_like(mask_binary)
    cv2.drawContours(solid_mask, [hull], -1, 255, -1)
    return solid_mask

def is_board_fully_visible(mask, frame_w, frame_h, margin=15):
    if np.count_nonzero(mask) == 0: return False
    x, y, w, h = cv2.boundingRect(mask)
    return (x > margin) and (y > margin) and (x + w < frame_w - margin) and (y + h < frame_h - margin)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmin(diff)] # BL
    return rect

def get_board_corners(mask_binary):
    solid = get_solid_mask(mask_binary)
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    rect = cv2.minAreaRect(largest)
    return np.int32(cv2.boxPoints(rect))

def get_birdseye_view(img, corners, size=640, margin_ratio=0.15):
    rect = order_points(corners)
    margin = int(size * margin_ratio)
    dst = np.array([
        [margin, margin], [size - margin - 1, margin],
        [size - margin - 1, size - margin - 1], [margin, size - margin - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (size, size)), M

def find_extreme_corners(feet_points, warped_mask):
    """Finds extreme pieces only if they lie within the solid board area."""
    solid_mask = get_solid_mask(warped_mask)
    valid_feet = [p for p in feet_points if 0 <= int(p[0]) < solid_mask.shape[1] 
                  and 0 <= int(p[1]) < solid_mask.shape[0] 
                  and solid_mask[int(p[1]), int(p[0])] > 0]
    
    if len(valid_feet) < 4: return None
    f_np = np.array(valid_feet)
    tl = f_np[np.argmin(f_np[:, 0] + f_np[:, 1])]
    br = f_np[np.argmax(f_np[:, 0] + f_np[:, 1])]
    tr = f_np[np.argmax(f_np[:, 0] - f_np[:, 1])]
    bl = f_np[np.argmin(f_np[:, 0] - f_np[:, 1])]
    return tl, tr, bl, br

def update_board_state_sam3(warped_img, image_model, warped_mask):
    vis_img = warped_img.copy()
    board_state = np.zeros((8, 8), dtype=int)
    
    proc = Sam3Processor(image_model)
    inf_state = proc.set_image(Image.fromarray(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)))
    output = proc.set_text_prompt(state=inf_state, prompt="chess piece")
    
    boxes, scores = output["boxes"], output["scores"]
    mask_conf = (scores > 0.4).cpu().numpy()
    if mask_conf.sum() < 4: return board_state, vis_img

    valid_boxes = boxes[mask_conf].cpu().numpy()
    feet_list = [[(b[0]+b[2])/2, b[3]] for b in valid_boxes]

    extremes = find_extreme_corners(feet_list, warped_mask)
    if extremes:
        pt_tl, pt_tr, pt_bl, pt_br = extremes
        v_top, v_left = pt_tr - pt_tl, pt_bl - pt_tl
        for box, feet in zip(valid_boxes, feet_list):
            rel = np.array(feet) - pt_tl
            col, row = int((rel[0] / (v_top[0] + 1e-6)) * 8), int((rel[1] / (v_left[1] + 1e-6)) * 8)
            board_state[np.clip(row, 0, 7), np.clip(col, 0, 7)] = 1
            cv2.rectangle(vis_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    return board_state, vis_img

# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------

device, dtype = "cuda", torch.bfloat16
image_model = build_sam3_image_model()
tracker_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device).eval()
tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

cap = cv2.VideoCapture("test1.mp4")
ret, first_frame = cap.read()
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_tracked.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (w, h))

# --- INITIAL DETECTION ---
init_proc = Sam3Processor(image_model)
inf_state = init_proc.set_image(Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)))
out_init = init_proc.set_text_prompt(state=inf_state, prompt="chessboard")
best_mask = out_init["masks"][out_init["scores"].argmax()].cpu().numpy().squeeze()
mask_np = (best_mask * 255).astype(np.uint8)
y_c, x_c = np.where(mask_np > 0)
points = [[int(x_c[i]), int(y_c[i])] for i in np.linspace(0, len(x_c)-1, 16, dtype=int)]

# --- INIT TRACKER SESSION ---
session = tracker_processor.init_video_session(inference_device=device, dtype=dtype)
inputs0 = tracker_processor(images=first_frame, return_tensors="pt").to(device, dtype=dtype)
tracker_processor.add_inputs_to_inference_session(
    session, frame_idx=0, obj_ids=1, input_points=[[points]], input_labels=[[[1]*len(points)]], original_size=inputs0.original_sizes[0]
)
with torch.no_grad(): tracker_model(inference_session=session, frame=inputs0.pixel_values[0])

# --- LOOP ---
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    inputs = tracker_processor(images=frame, return_tensors="pt").to(device, dtype=dtype)
    with torch.no_grad(): 
        output = tracker_model(inference_session=session, frame=inputs.pixel_values[0])
    
    masks_out = tracker_processor.post_process_masks([output.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
    current_mask = masks_out[0, 0].cpu().numpy().astype(np.uint8) * 255

    overlay = frame.copy()
    corners = get_board_corners(current_mask)
    
    if corners is not None:
        cv2.polylines(overlay, [corners], True, (0, 255, 255), 3)
        warped_board, M = get_birdseye_view(frame, corners)
        warped_mask = cv2.warpPerspective(current_mask, M, (640, 640))
        
        if is_board_fully_visible(current_mask, w, h):
            grid, vis = update_board_state_sam3(warped_board, image_model, warped_mask)
            if frame_idx % 10 == 0: print(f"Frame {frame_idx} State:\n{grid}")

    out.write(overlay)

cap.release()
out.release()

# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor
# from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# # ---------------------------------------------
# # HELPER FUNCTIONS
# # ---------------------------------------------
# def is_board_fully_visible(mask, frame_w, frame_h, margin=10):
#     if np.count_nonzero(mask) == 0:
#         return False
#     x, y, w, h = cv2.boundingRect(mask)
#     x2 = x + w
#     y2 = y + h
#     if (x > margin) and (y > margin) and (x2 < frame_w - margin) and (y2 < frame_h - margin):
#         return True
#     return False

# def order_points(pts):
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
#     largest = max(contours, key=cv2.contourArea)
#     hull = cv2.convexHull(largest)
#     perimeter = cv2.arcLength(hull, True)
#     for eps_factor in np.linspace(0.02, 0.1, 10):
#         epsilon = eps_factor * perimeter
#         approx = cv2.approxPolyDP(hull, epsilon, True)
#         if len(approx) == 4:
#             return approx.reshape(4, 2)
#     rect = cv2.minAreaRect(largest)
#     box = cv2.boxPoints(rect)
#     return np.int32(box)

# def get_birdseye_view(img, corners, size=640, margin_ratio=0.15):
#     rect = order_points(corners)
#     margin = int(size * margin_ratio)
#     dst = np.array([
#         [margin, margin],
#         [size - margin - 1, margin],
#         [size - margin - 1, size - margin - 1],
#         [margin, size - margin - 1]
#     ], dtype="float32")
#     M = cv2.getPerspectiveTransform(rect, dst)
#     return cv2.warpPerspective(img, M, (size, size)), M

# def is_point_in_mask(point, mask, margin=5):
#     x, y = int(round(point[0])), int(round(point[1]))
#     h, w = mask.shape[:2]
#     if x < margin or x >= w - margin or y < margin or y >= h - margin:
#         return False
#     region = mask[max(0, y-margin):min(h, y+margin+1), 
#                   max(0, x-margin):min(w, x+margin+1)]
#     return np.any(region > 0)

# def find_extreme_corners_in_mask(feet_points, img_w, img_h, warped_mask):
#     """
#     Filters points to keep ONLY those inside the mask, then finds corners.
#     """
#     # 1. Filter valid points first
#     valid_feet = []
#     for pt in feet_points:
#         if is_point_in_mask(pt, warped_mask):
#             valid_feet.append(pt)
            
#     if len(valid_feet) < 4:
#         return None

#     # 2. Find extremes among VALID points
#     feet_np = np.array(valid_feet)
#     tl_idx = np.argmin(feet_np[:, 0] + feet_np[:, 1])
#     br_idx = np.argmax(feet_np[:, 0] + feet_np[:, 1])
#     tr_idx = np.argmax(feet_np[:, 0] - feet_np[:, 1])
#     bl_idx = np.argmin(feet_np[:, 0] - feet_np[:, 1])
    
#     return feet_np[tl_idx], feet_np[tr_idx], feet_np[bl_idx], feet_np[br_idx]

# def interpolate_grid_points(p1, p2, steps=8):
#     return np.linspace(p1, p2, steps + 1)

# def update_board_state_sam3(warped_img, image_model, warped_mask):
#     img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(img_rgb)
#     vis_img = warped_img.copy()
#     h, w = warped_img.shape[:2]
#     board_state = np.zeros((8, 8), dtype=int)
    
#     # --- FILL HOLES IN MASK ---
#     cnts, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if cnts:
#         largest_cnt = max(cnts, key=cv2.contourArea)
#         cv2.drawContours(warped_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
#     # --------------------------

#     # 1. Detect Pieces
#     prompts = [("chess piece", 1)]
#     all_boxes, all_scores, all_colors = [], [], []
    
#     processor = Sam3Processor(image_model)
#     inference_state = processor.set_image(pil_img)
    
#     for prompt_text, color_id in prompts:
#         output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
#         boxes = output["boxes"]
#         scores = output["scores"]
#         high_conf_indices = scores > 0.4
        
#         if high_conf_indices.sum() > 0:
#             all_boxes.append(boxes[high_conf_indices])
#             all_scores.append(scores[high_conf_indices])
#             all_colors.extend([color_id] * len(boxes[high_conf_indices]))

#     if len(all_boxes) == 0:
#         return board_state, vis_img

#     all_boxes = torch.cat(all_boxes, dim=0) if isinstance(all_boxes[0], torch.Tensor) else np.vstack(all_boxes)
#     if isinstance(all_boxes, torch.Tensor): all_boxes = all_boxes.cpu().numpy()
    
#     feet_list = []
#     for box in all_boxes:
#         x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
#         feet_x, feet_y = (x1 + x2) / 2, y2
#         feet_list.append([feet_x, feet_y])

#     # 2. Determine Grid
#     corner_result = find_extreme_corners_in_mask(feet_list, w, h, warped_mask)
    
#     if corner_result is not None:
#         pt_tl, pt_tr, pt_bl, pt_br = corner_result
        
#         # --- DRAW GRID ---
#         top_edge = interpolate_grid_points(pt_tl, pt_tr)
#         bottom_edge = interpolate_grid_points(pt_bl, pt_br)
#         left_edge = interpolate_grid_points(pt_tl, pt_bl)
#         right_edge = interpolate_grid_points(pt_tr, pt_br)

#         for i in range(9):
#             cv2.line(vis_img, tuple(top_edge[i].astype(int)), tuple(bottom_edge[i].astype(int)), (255, 0, 0), 2)
#             cv2.line(vis_img, tuple(left_edge[i].astype(int)), tuple(right_edge[i].astype(int)), (255, 0, 0), 2)

#         vec_top_x = pt_tr[0] - pt_tl[0]
#         vec_left_y = pt_bl[1] - pt_tl[1]
        
#         # --- MAP PIECES & DRAW VALID ONES ---
#         for i, (box, feet, color_id) in enumerate(zip(all_boxes, feet_list, all_colors)):
#             # Check if valid (inside mask)
#             if is_point_in_mask(feet, warped_mask):
#                 fx, fy = feet
#                 rel_x = fx - pt_tl[0]
#                 rel_y = fy - pt_tl[1]
#                 norm_x = rel_x / (vec_top_x + 1e-6)
#                 norm_y = rel_y / (vec_left_y + 1e-6)
#                 col = int(norm_x * 8)
#                 row = int(norm_y * 8)
                
#                 board_state[np.clip(row, 0, 7), np.clip(col, 0, 7)] = color_id
                
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
#                 cv2.circle(vis_img, (int(fx), int(fy)), 5, (0,255,0), -1) # Green dot
#                 cv2.putText(vis_img, f"{row},{col}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
#             else:
#                 # INVALID: Skip drawing entirely
#                 pass
#     else:
#         print("Warning: Could not find 4 valid corners inside mask.")
#         cv2.putText(vis_img, "NO VALID GRID", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     return board_state, vis_img

# # ---------------------------------------------
# # MAIN PIPELINE
# # ---------------------------------------------
# device = "cuda"
# dtype = torch.bfloat16
# MAX_FRAMES = None
# DETECTION_PROMPT = "chessboard"
# WARP_SIZE = 640
# FRAME_MARGIN_THRESHOLD = 15

# print("Loading Models...")
# image_model = build_sam3_image_model()
# tracker_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device).eval()
# tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

# cap = cv2.VideoCapture("test1.mp4")
# ret, first_frame = cap.read()
# if not ret:
#     print("Error reading video")
#     exit()

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('output_tracked.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# # --- INITIAL DETECTION ---
# print("Initial Detection...")
# init_proc = Sam3Processor(image_model)
# inf_state = init_proc.set_image(Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)))
# out_init = init_proc.set_text_prompt(state=inf_state, prompt=DETECTION_PROMPT)

# best_mask = out_init["masks"][out_init["scores"].argmax()]
# if isinstance(best_mask, torch.Tensor): best_mask = best_mask.cpu().numpy()
# if best_mask.ndim > 2: best_mask = best_mask.squeeze()
# mask_np = (best_mask * 255).astype(np.uint8) if best_mask.max() <= 1.0 else best_mask.astype(np.uint8)
# mask_np = cv2.resize(mask_np, (width, height))

# y_c, x_c = np.where(mask_np > 0)
# points = [[int(x_c[i]), int(y_c[i])] for i in np.linspace(0, len(x_c)-1, 16, dtype=int)]

# # --- INIT TRACKER ---
# session = tracker_processor.init_video_session(inference_device=device, dtype=dtype)
# inputs0 = tracker_processor(images=first_frame, return_tensors="pt").to(device, dtype=dtype)
# tracker_processor.add_inputs_to_inference_session(
#     inference_session=session, frame_idx=0, obj_ids=1,
#     input_points=[[points]], input_labels=[[[1]*len(points)]], original_size=inputs0.original_sizes[0]
# )
# with torch.no_grad(): tracker_model(inference_session=session, frame=inputs0.pixel_values[0])

# # --- LOOP ---
# frame_idx = 0
# while True:
#     if MAX_FRAMES and frame_idx >= MAX_FRAMES: break
#     ret, frame = cap.read()
#     if not ret: break
#     frame_idx += 1

#     inputs = tracker_processor(images=frame, return_tensors="pt").to(device, dtype=dtype)
#     with torch.no_grad(): output = tracker_model(inference_session=session, frame=inputs.pixel_values[0])
    
#     masks_out = tracker_processor.post_process_masks([output.pred_masks], original_sizes=[[height, width]], binarize=True)[0]
#     current_mask = masks_out[0, 0].cpu().numpy().astype(np.uint8) * 255

#     overlay = frame.copy()
#     corners = get_board_corners(current_mask)
    
#     if corners is not None:
#         cv2.polylines(overlay, [corners], True, (0, 255, 255), 3)
#         warped_board, M = get_birdseye_view(frame, corners, size=WARP_SIZE)
#         warped_mask = cv2.warpPerspective(current_mask, M, (WARP_SIZE, WARP_SIZE))
        
#         fully_visible = is_board_fully_visible(current_mask, width, height, margin=FRAME_MARGIN_THRESHOLD)

#         if frame_idx % 1 == 0:
#             if fully_visible:
#                 print(f"Frame {frame_idx}: Board fully visible. Detecting pieces & building grid...")
#                 grid_state, vis_img = update_board_state_sam3(warped_board, image_model, warped_mask)
                
#                 # SAVE THE IMAGE WITH GRID
#                 cv2.imwrite(f"state_debug_{frame_idx}.jpg", vis_img)
#                 print(f"--- Saved state_debug_{frame_idx}.jpg ---")
#             else:
#                 print(f"Frame {frame_idx}: Skipped (Clipped).")
#                 cv2.putText(overlay, "BOARD CLIPPED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#         if 'warped_board' in locals():
#             thumb = cv2.resize(warped_board, (200, 200))
#             overlay[0:200, 0:200] = thumb

#     out.write(overlay)
#     print(f"Processed frame {frame_idx}")

# cap.release()
# out.release()
# print("Done!")