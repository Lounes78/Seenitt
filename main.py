import argparse
import cv2
from ultralytics import YOLO
import numpy as np
from card import Card
from hand import Hand


def load_model(weights_path: str) -> YOLO:
  return YOLO(weights_path)


def init_writer_from_frame(frame, fps: float, output_path: str | None):
  """Initialize a VideoWriter using the first frame shape to avoid 0-size writer."""
  if not output_path:
    return None
  height, width = frame.shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def process_simple(model: YOLO, video_path: str, output_path: str | None) -> None:
  capture = cv2.VideoCapture(video_path)
  if not capture.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

  fps = capture.get(cv2.CAP_PROP_FPS) or 30
  writer = None

  while capture.isOpened():
    success, frame = capture.read()
    if not success:
      break

    if writer is None:
      writer = init_writer_from_frame(frame, fps, output_path)

    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()

    if writer:
      writer.write(annotated_frame)

    cv2.imshow('Cards', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  capture.release()
  if writer:
    writer.release()
  cv2.destroyAllWindows()


def process_blackjack(model: YOLO, video_path: str, output_path: str | None) -> None:
  capture = cv2.VideoCapture(video_path)
  if not capture.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

  fps = capture.get(cv2.CAP_PROP_FPS) or 30
  width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  writer = None

  while capture.isOpened():
    success, frame = capture.read()
    if not success:
      break

    if writer is None:
      writer = init_writer_from_frame(frame, fps, output_path)

    results = model.track(frame, persist=True, verbose=False)
    if len(results) == 0:
      continue

    annotated_frame = results[0].plot()
    result = results[0].cpu().boxes

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

    cv2.imshow('Cards', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  capture.release()
  if writer:
    writer.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run YOLO detection or blackjack helper on a video file.')
  parser.add_argument('video', help='Path to input video file')
  parser.add_argument('--output', '-o', help='Path to save annotated video (mp4)')
  parser.add_argument('--mode', choices=['blackjack', 'simple'], default='blackjack', help='simple = boxes only, blackjack = strategy overlay')
  parser.add_argument('--weights', default='runs/detect/train7/weights/best.pt', help='Path to YOLO weights')
  args = parser.parse_args()

  model = load_model(args.weights)
  if args.mode == 'simple':
    process_simple(model, args.video, args.output)
  else:
    process_blackjack(model, args.video, args.output)