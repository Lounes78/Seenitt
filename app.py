import argparse
from main import load_model, process_simple, process_blackjack


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Alias wrapper; prefer running main.py directly.')
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
  args = parser.parse_args()

  model = load_model(args.weights, device=args.device, half=args.half)
  if args.mode == 'simple':
    process_simple(model, args.video, args.output, display=not args.no_display, imgsz=args.imgsz, log_every=args.log_every)
  else:
    process_blackjack(model, args.video, args.output, display=not args.no_display, imgsz=args.imgsz, use_track=args.track, log_every=args.log_every)
