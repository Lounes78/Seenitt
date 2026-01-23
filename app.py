import os
import argparse
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import numpy as np

load_dotenv()

# rf = Roboflow(api_key=os.getenv('API_KEY', ''))
# project = rf.workspace().project("playing-cards-ow27d")
# model = project.version(4).model

def run_video(video_path: str) -> None:
  """Run YOLO inference on a video file frame by frame."""
  model = YOLO('runs/detect/train7/weights/best.pt')

  capture = cv2.VideoCapture(video_path)
  if not capture.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

  while True:
    success, frame = capture.read()
    if not success:
      break

    predictions = model.predict(frame, verbose=False)

    for r in predictions:
      annotator = Annotator(frame)
      for box in r.boxes:
        b = box.xyxy[0]
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
      frame = annotator.result()

    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  capture.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run YOLOv8 detection on a video file.')
  parser.add_argument('video', help='Path to the input video file')
  args = parser.parse_args()

  run_video(args.video)
