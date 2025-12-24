import torch
import cv2
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# Configuration
device = "cuda"
dtype = torch.bfloat16
MAX_FRAMES = 500  
DETECTION_PROMPT = "chessboard"  

# Load models
print("Loading SAM3 image model for detection...")
image_model = build_sam3_image_model()
image_processor = Sam3Processor(image_model)

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

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (width, height))

# ---------------------------------------------
# DETECT CHESSBOARD using SAM3 image model
# ---------------------------------------------
print(f"Detecting chessboard with prompt: '{DETECTION_PROMPT}'")

# Convert frame to PIL Image
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(first_frame_rgb)

# Use SAM3 image model to detect chessboard
inference_state = image_processor.set_image(pil_image)
output = image_processor.set_text_prompt(state=inference_state, prompt=DETECTION_PROMPT)

# Get the masks, bounding boxes, and scores
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

print(f"Found {len(masks)} detections")
print(f"Scores: {scores}")

if len(masks) == 0:
    raise ValueError(f"No chessboard detected with prompt '{DETECTION_PROMPT}'. Try a different prompt.")

# Use the highest scoring detection
best_idx = scores.argmax()
best_mask = masks[best_idx]
best_box = boxes[best_idx]
best_score = scores[best_idx]

print(f"Using detection with score: {best_score:.3f}")
print(f"Bounding box: {best_box}")

# Visualize detection
vis_frame = first_frame.copy()
x1, y1, x2, y2 = map(int, best_box)
cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Convert mask to numpy if needed
if isinstance(best_mask, torch.Tensor):
    mask_np = best_mask.cpu().numpy()
else:
    mask_np = best_mask

# Ensure mask is 2D (remove any extra dimensions)
if mask_np.ndim > 2:
    mask_np = mask_np.squeeze()

# Resize mask to match frame dimensions if needed
if mask_np.shape != (first_frame.shape[0], first_frame.shape[1]):
    mask_np = cv2.resize(mask_np, (first_frame.shape[1], first_frame.shape[0]))

# Overlay mask
if mask_np.max() <= 1.0:
    mask_np = (mask_np * 255).astype(np.uint8)
else:
    mask_np = mask_np.astype(np.uint8)

# Create mask for overlay
mask_bool = mask_np > 0
vis_frame[mask_bool] = vis_frame[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
cv2.imwrite("chessboard_detection.jpg", vis_frame)
print("Saved detection visualization to chessboard_detection.jpg")

# Extract points from the mask for tracking
# Sample points from the detected mask region
y_coords, x_coords = np.where(mask_np > 0)
if len(x_coords) == 0:
    raise ValueError("Detected mask is empty")

# Sample points evenly across the mask
num_points = 16
indices = np.linspace(0, len(x_coords) - 1, num_points, dtype=int)
points = [[int(x_coords[i]), int(y_coords[i])] for i in indices]

print(f"Extracted {len(points)} points from detected mask for tracking")

# Visualize points
vis_points = first_frame.copy()
for pt in points:
    cv2.circle(vis_points, tuple(pt), 5, (0, 0, 255), -1)
cv2.imwrite("tracking_points.jpg", vis_points)
print("Saved tracking points to tracking_points.jpg")

# ---------------------------------------------
# Init tracking session with SAM3 tracker
# ---------------------------------------------
print("Initializing video tracking session...")
session = tracker_processor.init_video_session(
    inference_device=device,
    dtype=dtype,
)

# ---------------------------------------------
# FRAME 0 — add prompts and segment
# ---------------------------------------------
inputs0 = tracker_processor(
    images=first_frame,
    return_tensors="pt",
).to(device, dtype=dtype)

# Format points for SAM3 tracker: [[[point1, point2, ...]]]
formatted_points = [[[pt for pt in points]]]
formatted_labels = [[[1] * len(points)]]

# ADD PROMPT
tracker_processor.add_inputs_to_inference_session(
    inference_session=session,
    frame_idx=0,
    obj_ids=1,
    input_points=formatted_points,
    input_labels=formatted_labels,
    original_size=inputs0.original_sizes[0],
)

# Segment on frame 0
with torch.no_grad():
    output0 = tracker_model(
        inference_session=session,
        frame=inputs0.pixel_values[0],
    )
    
    # Post-process to visualize initial segmentation
    masks0 = tracker_processor.post_process_masks(
        [output0.pred_masks], 
        original_sizes=[[first_frame.shape[0], first_frame.shape[1]]], 
        binarize=True
    )[0]
    
    # Overlay mask on first frame
    mask_np_track = masks0[0, 0].cpu().numpy().astype(np.uint8) * 255
    overlay = first_frame.copy()
    overlay[mask_np_track > 0] = overlay[mask_np_track > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imwrite("chessboard_mask_frame0.jpg", overlay)
    
    # Write first frame to video
    out.write(overlay.astype(np.uint8))
    
    print("Saved initial tracking segmentation to chessboard_mask_frame0.jpg")

print("✓ Initialized chessboard tracking on frame 0")

# ---------------------------------------------
# TRACKING on remaining frames
# ---------------------------------------------
frame_idx = 1
fps_times = []

with torch.no_grad():
    while True:
        # Check frame limit
        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            print(f"Reached maximum frame limit: {MAX_FRAMES}")
            break
        
        ret, frame = cap.read()
        if not ret:
            break

        inputs = tracker_processor(
            images=frame,
            return_tensors="pt",
        ).to(device, dtype=dtype)

        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)

        start.record()
        output = tracker_model(
            inference_session=session,
            frame=inputs.pixel_values[0],
        )
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        fps_times.append(elapsed)
        
        # Post-process and create overlay
        masks = tracker_processor.post_process_masks(
            [output.pred_masks], 
            original_sizes=[[frame.shape[0], frame.shape[1]]], 
            binarize=True
        )[0]
        mask_np = masks[0, 0].cpu().numpy().astype(np.uint8) * 255
        overlay = frame.copy()
        overlay[mask_np > 0] = overlay[mask_np > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        # Write frame to video
        out.write(overlay.astype(np.uint8))
        
        print(f"Frame {frame_idx}: {elapsed:.2f} ms ({1000/elapsed:.1f} FPS)")
        
        # Save specific frames as images
        if frame_idx % 30 == 0:  # Save every 30 frames
            cv2.imwrite(f"chessboard_mask_frame{frame_idx}.jpg", overlay)

        frame_idx += 1

cap.release()
out.release()

# Statistics
print(f"\n{'='*50}")
print(f"Processed {frame_idx} frames")
print(f"Average: {np.mean(fps_times):.2f} ms ({1000/np.mean(fps_times):.1f} FPS)")
print(f"Min: {np.min(fps_times):.2f} ms ({1000/np.min(fps_times):.1f} FPS)")
print(f"Max: {np.max(fps_times):.2f} ms ({1000/np.max(fps_times):.1f} FPS)")
print(f"Output video saved to: output_tracked.mp4")
print(f"{'='*50}")
