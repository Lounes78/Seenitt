from ultralytics import YOLO
import cv2
import sys
import math

# ===================== CONFIG =====================
MODEL_PATH = "yolov8s_playing_cards.pt"
IMG_SIZE = 960
CONF = 0.2
IOU = 0.4
DIST_THRESHOLD = 120

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else None
OUTPUT_VIDEO = "result_clean_video.mp4"
# =================================================

if VIDEO_PATH is None:
    print("Usage: python yolo_cards_video_headless.py <video.mp4>")
    sys.exit(1)

model = YOLO(MODEL_PATH)

# ===================== VIDEO IO =====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la vidéo")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
# ===================================================


# ===================== UTILITAIRES =====================
def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c2[1] - c1[1])
# =====================================================


# ===================== LOOP VIDEO =====================
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        verbose=False
    )

    detections = []

    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            detections.append({
                "label": model.names[int(b.cls)],
                "conf": float(b.conf),
                "box": (x1, y1, x2, y2),
                "center": center((x1, y1, x2, y2))
            })

    groups = []

    for det in detections:
        placed = False
        for g in groups:
            if distance(det["center"], g["center"]) < DIST_THRESHOLD:
                g["items"].append(det)
                placed = True
                break
        if not placed:
            groups.append({
                "center": det["center"],
                "items": [det]
            })

    final_cards = []
    for g in groups:
        best = max(g["items"], key=lambda x: x["conf"])
        final_cards.append(best)

    for card in final_cards:
        x1, y1, x2, y2 = card["box"]
        label = card["label"]
        conf = card["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    writer.write(frame)
# =====================================================


cap.release()
writer.release()

print("Vidéo traitée avec succès (mode headless)")
print("Fichier de sortie :", OUTPUT_VIDEO)
