import cv2
from PIL import Image
from model_handler import PlantIDModel
from manager import CatalogManager
from quality import sharpness_score, frame_difference_score
from time import time
from datetime import datetime
import numpy as np

FRAME_SHARPNESS_THRESHOLD = 400
FRAME_DIFF_THRESHOLD = 18.0
PROCESS_EVERY_N_FRAMES = 10


# ------------------------------------------------------------
# Highlight overlay helper
# ------------------------------------------------------------

class HighlightEvent:
    def __init__(self, box, label, obj_id, snapshot, start_frame, duration=30):
        self.box = box
        self.label = label
        self.obj_id = obj_id
        self.snapshot = snapshot
        self.duration = duration
        self.start_frame = start_frame
        self.end_frame = start_frame + duration

    def refresh(self, box, snapshot, frame_idx, duration=30):
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

        color = (0, 255, 255) if (frame_idx - hlt.start_frame) < 5 else (0, 200, 0)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        label = f"{hlt.label} (ID {hlt.obj_id})"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(video_path):
    model = PlantIDModel(model_paths={})
    catalog = CatalogManager()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"output_seenit_{ts}.mp4"

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    prev_frame_pil = None
    active_highlights = {}
    last_processed_frame_idx = -PROCESS_EVERY_N_FRAMES
    
    while cap.isOpened():
        start_time = time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # ----------------------------------------------------
        # Decide whether to process this frame
        # ----------------------------------------------------

        frame_gap = frame_idx - last_processed_frame_idx
        do_process = frame_gap >= PROCESS_EVERY_N_FRAMES
        
        if do_process:
            sharp = sharpness_score(frame_pil)
            if sharp < FRAME_SHARPNESS_THRESHOLD:
                do_process = False
            elif prev_frame_pil is not None:
                diff = frame_difference_score(prev_frame_pil, frame_pil)
                if diff < FRAME_DIFF_THRESHOLD:
                    do_process = False

        # ----------------------------------------------------
        # Detection + catalog
        # ----------------------------------------------------

        if do_process:
            last_processed_frame_idx = frame_idx
            detections = model.process_frame(frame_pil, prompt="tree")

            for det in detections:
                new_id = catalog.process_detection(
                    crop=det["crop"],
                    frame=frame_pil,
                    label=det["label"],
                    confidence=det["confidence"],
                    box=det["box"],
                    frame_idx=frame_idx,
                )

                if new_id is not None:
                    if new_id not in active_highlights:
                        active_highlights[new_id] = HighlightEvent(
                            box=det["box"],
                            label=det["label"],
                            obj_id=new_id,
                            snapshot=det["crop"],
                            start_frame=frame_idx,
                            duration=30,
                        )
                    else:
                        # 🔁 même ID → on prolonge seulement l’affichage
                        hlt = active_highlights[new_id]
                        hlt.start_frame = frame_idx
                        hlt.end_frame = frame_idx + hlt.duration


        # ----------------------------------------------------
        # Draw highlights AFTER detection
        # ----------------------------------------------------

        frame, active_highlights = draw_highlights(
            frame, active_highlights, frame_idx
        )


        active_highlights = {
            obj_id: hlt
            for obj_id, hlt in active_highlights.items()
            if frame_idx <= hlt.end_frame
        }


        out.write(frame)

        prev_frame_pil = frame_pil

        if do_process:
            print(
                f"Frame {frame_idx} | processed={do_process} | "
                f"time={time() - start_time:.3f}s"
            )

    out.release()
    cap.release()
    print(f"✅ Output saved to {out_path}")


if __name__ == "__main__":
    main("/home/camembert/PFE-G11-SeenIt/data/test_video.mp4")