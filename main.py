import argparse
import os
import time
import cv2
from ultralytics import YOLO
import numpy as np
from card import Card
from hand import Hand


def load_model(weights_path: str, device: str | None = None, half: bool = False) -> YOLO:
  """Load YOLO model on selected device. Optionally cast to half precision for GPU."""
  model = YOLO(weights_path, task='detect').to(device or 'cpu')
  if half and (device or 'cpu') != 'cpu':
    try:
      model.fuse()  # fuse conv+bn before casting to half to avoid dtype mismatch
      model.model.half()
    except Exception:
      pass
  return model


def init_writer_from_frame(frame, fps: float, output_path: str | None):
  """Initialize a VideoWriter using the first frame shape to avoid 0-size writer."""
  if not output_path:
    return None
  height, width = frame.shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def process_simple(model: YOLO, video_path: str, output_path: str | None, display: bool = True, imgsz: int = 640, log_every: int = 30, save_frames_dir: str | None = None) -> None:
  capture = cv2.VideoCapture(video_path)
  if not capture.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

  fps = capture.get(cv2.CAP_PROP_FPS) or 30
  writer = None

  if save_frames_dir:
    os.makedirs(save_frames_dir, exist_ok=True)

  frame_count = 0
  total_time = 0.0

  while capture.isOpened():
    success, frame = capture.read()
    if not success:
      break

    if writer is None:
      writer = init_writer_from_frame(frame, fps, output_path)

    start = time.time()
    results = model.predict(frame, verbose=False, imgsz=imgsz)
    annotated_frame = results[0].plot()
    elapsed = time.time() - start
    total_time += elapsed
    frame_count += 1

    if writer:
      writer.write(annotated_frame)

    if save_frames_dir:
      frame_path = os.path.join(save_frames_dir, f"frame_{frame_count:06d}.jpg")
      cv2.imwrite(frame_path, annotated_frame)

    if display:
      cv2.imshow('Cards', annotated_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if log_every and frame_count % log_every == 0:
      fps_now = frame_count / total_time if total_time else 0
      print(f"[simple] frames={frame_count}, avg_ms={total_time/frame_count*1000:.1f}, fps={fps_now:.1f}")

  capture.release()
  if writer:
    writer.release()
  cv2.destroyAllWindows()
  if frame_count:
    fps_now = frame_count / total_time if total_time else 0
    print(f"[simple] done: frames={frame_count}, avg_ms={total_time/frame_count*1000:.1f}, fps={fps_now:.1f}")


def process_blackjack(model: YOLO, video_path: str, output_path: str | None, display: bool = True, imgsz: int = 640, use_track: bool = False, log_every: int = 30, save_frames_dir: str | None = None) -> None:
  capture = cv2.VideoCapture(video_path)
  if not capture.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

  fps = capture.get(cv2.CAP_PROP_FPS) or 30
  width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  writer = None

  if save_frames_dir:
    os.makedirs(save_frames_dir, exist_ok=True)

  frame_count = 0
  total_time = 0.0

  while capture.isOpened():
    success, frame = capture.read()
    if not success:
      break

    if writer is None:
      writer = init_writer_from_frame(frame, fps, output_path)

    start = time.time()
    if use_track:
      results = model.track(frame, persist=True, verbose=False, imgsz=imgsz)
    else:
      results = model.predict(frame, verbose=False, imgsz=imgsz)
    if len(results) == 0:
      continue

    annotated_frame = results[0].plot()
    result = results[0].cpu().boxes
    elapsed = time.time() - start
    total_time += elapsed
    frame_count += 1

    names = model.names
    detect_xyxy = result.xyxy.tolist() if result.xyxy is not None else []

    class_bbox_dict = {}
    for c, bbox in zip(result.cls, detect_xyxy):
      class_id = int(c)
      class_name = names[class_id]
      if class_name not in class_bbox_dict:
        class_bbox_dict[class_name] = []
      x1, y1, x2, y2 = bbox
      top_left = {'x': x1, 'y': y1}
      bottom_right = {'x': x2, 'y': y2}
      card = Card(class_name, top_left, bottom_right)
      class_bbox_dict[class_name].append(card)

    dealer = None
    current_hand = Hand([])
    for _, positions in class_bbox_dict.items():
      if len(positions) > 0:
        for position in enumerate(positions):
          current_hand.add_card(position[1])
          if dealer is not None:
            if position[1].top_left['y'] < dealer.top_left['y']:
              current_hand.add_card(dealer)
              current_hand.remove_card(position[1])
              dealer = position[1]
            for card_in_hand in list(current_hand.cards):
              if card_in_hand.card_string == dealer.card_string:
                current_hand.remove_card(card_in_hand)
          else:
            current_hand.remove_card(position[1])
            dealer = position[1]

    card_list = []
    for card in current_hand.cards:
      if card.card_string not in [card.card_string for card in card_list]:
        card_list.append(card)
    card_list.sort(key=lambda card: card.top_left['x'])

    proximity_threshold = width / 4 if width else 0
    grouped_hands = []
    assigned = set()
    for card in card_list:
      if card in assigned:
        continue
      current_hand = Hand([card])
      assigned.add(card)
      for other_card in card_list:
        if other_card in assigned:
          continue
        if abs(card.top_left['x'] - other_card.top_left['x']) < proximity_threshold:
          current_hand.add_card(other_card)
          assigned.add(other_card)
      grouped_hands.append(current_hand)
      
    for hand in grouped_hands:
      cards = hand.cards
      x_coords = [card.top_left['x'] for card in cards]
      y_coords = [card.top_left['y'] for card in cards]

      x_min = min(x_coords)
      y_min = min(y_coords)
      x_max = max([card.bottom_right['x'] for card in cards])
      y_max = max([card.bottom_right['y'] for card in cards])

      cv2.rectangle(annotated_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
      if dealer:
        cv2.putText(annotated_frame, hand.calculate_best_action(dealer), (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if writer:
      writer.write(annotated_frame)

    if save_frames_dir:
      frame_path = os.path.join(save_frames_dir, f"frame_{frame_count:06d}.jpg")
      cv2.imwrite(frame_path, annotated_frame)

    if display:
      cv2.imshow('Cards', annotated_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if log_every and frame_count % log_every == 0:
      fps_now = frame_count / total_time if total_time else 0
      print(f"[blackjack] frames={frame_count}, avg_ms={total_time/frame_count*1000:.1f}, fps={fps_now:.1f}")

  capture.release()
  if writer:
    writer.release()
  cv2.destroyAllWindows()
  if frame_count:
    fps_now = frame_count / total_time if total_time else 0
    print(f"[blackjack] done: frames={frame_count}, avg_ms={total_time/frame_count*1000:.1f}, fps={fps_now:.1f}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run YOLO detection or blackjack helper on a video file.')
  parser.add_argument('video', help='Path to input video file')
  parser.add_argument('--output', '-o', help='Path to save annotated video (mp4)')
  parser.add_argument('--mode', choices=['blackjack', 'simple'], default='blackjack', help='simple = boxes only, blackjack = strategy overlay')
  parser.add_argument('--weights', default='runs/detect/train7/weights/best.pt', help='Path to YOLO weights')
  parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (short side). Lower is faster.')
  parser.add_argument('--device', default=None, help='Device: cpu, cuda, mps, or leave blank for auto/cpu')
  parser.add_argument('--track', action='store_true', help='Use tracking (slower, persistent IDs). Default off for speed.')
  parser.add_argument('--half', action='store_true', help='Use FP16 (GPU only) for speed.')
  parser.add_argument('--log-every', type=int, default=30, help='Log perf stats every N frames (0 to disable).')
  parser.add_argument('--no-display', action='store_true', help='Disable window display (faster, for headless/save only)')
  parser.add_argument('--save-frames', help='Directory to save annotated frames instead of/alongside video')
  args = parser.parse_args()

  model = load_model(args.weights, device=args.device, half=args.half)
  if args.mode == 'simple':
    process_simple(model, args.video, args.output, display=not args.no_display, imgsz=args.imgsz, log_every=args.log_every, save_frames_dir=args.save_frames)
  else:
    process_blackjack(model, args.video, args.output, display=not args.no_display, imgsz=args.imgsz, use_track=args.track, log_every=args.log_every, save_frames_dir=args.save_frames)