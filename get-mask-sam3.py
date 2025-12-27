import torch
import cv2
import numpy as np
import os
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
VIDEO_PATH = "test2.mp4"
OUTPUT_DIR = "masked_frames_chessboard"
DETECTION_PROMPT = "chessboard"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Models...")
# We need image model for initial detection
image_model = build_sam3_image_model()
# We need tracker model for video tracking
tracker_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=DTYPE).to(DEVICE).eval()
tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret:
    print(f"Error reading video from {VIDEO_PATH}")
    exit()

height, width = first_frame.shape[:2]

# --- INITIAL DETECTION ---
print("Initial Detection on first frame...")
init_proc = Sam3Processor(image_model)
inf_state = init_proc.set_image(Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)))
out_init = init_proc.set_text_prompt(state=inf_state, prompt=DETECTION_PROMPT)

# Get the best mask from initial detection
best_mask = out_init["masks"][out_init["scores"].argmax()]
if isinstance(best_mask, torch.Tensor): best_mask = best_mask.cpu().numpy()
if best_mask.ndim > 2: best_mask = best_mask.squeeze()
mask_np = (best_mask * 255).astype(np.uint8) if best_mask.max() <= 1.0 else best_mask.astype(np.uint8)
mask_np = cv2.resize(mask_np, (width, height))

# Get points for tracking initialization (grid of points inside the mask)
y_c, x_c = np.where(mask_np > 0)
if len(x_c) == 0:
    print("No chessboard detected in first frame.")
    exit()
    
# Sample points from the mask to initialize the tracker
points = [[int(x_c[i]), int(y_c[i])] for i in np.linspace(0, len(x_c)-1, 16, dtype=int)]

# --- INIT TRACKER ---
session = tracker_processor.init_video_session(inference_device=DEVICE, dtype=DTYPE)
inputs0 = tracker_processor(images=first_frame, return_tensors="pt").to(DEVICE, dtype=DTYPE)
tracker_processor.add_inputs_to_inference_session(
    inference_session=session, frame_idx=0, obj_ids=1,
    input_points=[[points]], input_labels=[[[1]*len(points)]], original_size=inputs0.original_sizes[0]
)
# Process first frame
with torch.no_grad(): 
    tracker_model(inference_session=session, frame=inputs0.pixel_values[0])

# Save first frame result
masked_first_frame = cv2.bitwise_and(first_frame, first_frame, mask=mask_np)
x, y, w, h = cv2.boundingRect(mask_np)
if w > 0 and h > 0:
    masked_first_frame = masked_first_frame[y:y+h, x:x+w]
cv2.imwrite(os.path.join(OUTPUT_DIR, "frame_0000.jpg"), masked_first_frame)
print("Processed frame 0")

# --- LOOP ---
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    inputs = tracker_processor(images=frame, return_tensors="pt").to(DEVICE, dtype=DTYPE)
    with torch.no_grad(): 
        output = tracker_model(inference_session=session, frame=inputs.pixel_values[0])
    
    masks_out = tracker_processor.post_process_masks([output.pred_masks], original_sizes=[[height, width]], binarize=True)[0]
    current_mask = masks_out[0, 0].cpu().numpy().astype(np.uint8) * 255

    # Apply mask to frame: keep only chessboard, rest black
    masked_frame = cv2.bitwise_and(frame, frame, mask=current_mask)
    
    # Crop to bbox
    x, y, w, h = cv2.boundingRect(current_mask)
    if w > 0 and h > 0:
        masked_frame = masked_frame[y:y+h, x:x+w]

    # Save image
    output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(output_path, masked_frame)
    
    if frame_idx % 10 == 0:
        print(f"Processed frame {frame_idx}")

cap.release()
print(f"Done! All masked frames saved to {OUTPUT_DIR}")
