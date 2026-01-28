import os
import cv2
import time
import numpy as np
from PIL import Image
from datetime import datetime

from model_handler import PlantIDModel
from manager import CatalogManager
from quality import sharpness_score, frame_difference_score

# ============================================================
# CONFIG
# ============================================================

FRAME_SHARPNESS_THRESHOLD = 400
FRAME_DIFF_THRESHOLD = 18.0
PROCESS_EVERY_N_FRAMES = 10

ENABLE_VIDEO_OUTPUT = True
VIDEO_FPS = 25
VIDEO_CODEC = "mp4v"
OUTPUT_DIR = "stream_output"

# ============================================================
# Highlight overlay
# ============================================================

class HighlightEvent:
    def __init__(self, box, label, obj_id, snapshot, start_frame, duration=30):
        self.box = box
        self.label = label
        self.obj_id = obj_id
        self.snapshot = snapshot
        self.start_frame = start_frame
        self.end_frame = start_frame + duration
        self.duration = duration

    def refresh(self, box, snapshot, frame_idx, duration=30):
        # Unused
        self.box = box
        self.snapshot = snapshot
        self.start_frame = frame_idx
        self.end_frame = frame_idx + duration

def draw_highlights(frame_bgr, highlights, frame_idx):
    h, w = frame_bgr.shape[:2]
    remaining = {}

    for obj_id, hlt in highlights.items():
        if frame_idx > hlt.end_frame:
            continue

        x1, y1, x2, y2 = map(int, hlt.box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        bw, bh = x2 - x1, y2 - y1
        if bw <= 2 or bh <= 2:
            continue

        snap = hlt.snapshot.resize((bw, bh))
        snap_np = cv2.cvtColor(np.array(snap), cv2.COLOR_RGB2BGR)
        frame_bgr[y1:y2, x1:x2] = snap_np

        color = (0, 255, 255) if frame_idx - hlt.start_frame < 5 else (0, 200, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        label = f"{hlt.label} (ID {hlt.obj_id})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(
            frame_bgr,
            (x1, y1 - th - 6),
            (x1 + tw + 4, y1),
            color,
            -1,
        )

        cv2.putText(
            frame_bgr,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        remaining[obj_id] = hlt

    return frame_bgr, remaining


# ============================================================
# MAIN WORKER
# ============================================================

def plant_worker_process(frame_queue, result_queue=None):
    print("[PlantWorker] Starting...", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = PlantIDModel(model_paths={})
    catalog = CatalogManager()

    active_highlights = {}
    frame_idx = 0
    last_processed_frame_idx = -PROCESS_EVERY_N_FRAMES
    prev_frame_pil = None

    video_writer = None
    video_path = None

    while True:
        item = frame_queue.get()
        if item is None:
            break

        frame_idx, frame_bgr = item
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # --------------------------------------------
        # Decide whether to process
        # --------------------------------------------

        do_process = (frame_idx - last_processed_frame_idx) >= PROCESS_EVERY_N_FRAMES

        if do_process:
            sharp = sharpness_score(frame_pil)
            if sharp < FRAME_SHARPNESS_THRESHOLD:
                do_process = False
            elif prev_frame_pil is not None:
                diff = frame_difference_score(prev_frame_pil, frame_pil)
                if diff < FRAME_DIFF_THRESHOLD:
                    do_process = False

        # --------------------------------------------
        # Detection
        # --------------------------------------------

        if do_process:
            last_processed_frame_idx = frame_idx
            detections = model.process_frame(frame_pil, prompt="tree")

            for det in detections:
                obj_id = catalog.process_detection(
                    crop=det["crop"],
                    frame=frame_pil,
                    label=det["label"],
                    confidence=det["confidence"],
                    box=det["box"],
                    frame_idx=frame_idx,
                )

                if obj_id is not None and obj_id not in active_highlights:
                    active_highlights[obj_id] = HighlightEvent(
                        box=det["box"],
                        label=det["label"],
                        obj_id=obj_id,
                        snapshot=det["crop"],
                        start_frame=frame_idx,
                        duration=30,
                    )

        # --------------------------------------------
        # Draw highlights
        # --------------------------------------------

        frame_bgr, active_highlights = draw_highlights(
            frame_bgr, active_highlights, frame_idx
        )

        # --------------------------------------------
        # Lazy video init
        # --------------------------------------------

        if ENABLE_VIDEO_OUTPUT and video_writer is None:
            h, w = frame_bgr.shape[:2]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(OUTPUT_DIR, f"stream_{ts}.mp4")

            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                VIDEO_FPS,
                (w, h),
            )

            print(f"[PlantWorker] Video output → {video_path}", flush=True)

        if video_writer is not None:
            video_writer.write(frame_bgr)

        prev_frame_pil = frame_pil

    if video_writer is not None:
        video_writer.release()
        print(f"[PlantWorker] Video saved: {video_path}", flush=True)

    print("[PlantWorker] Shutdown clean", flush=True)
