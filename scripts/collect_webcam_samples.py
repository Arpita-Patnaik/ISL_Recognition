import argparse
import time
from pathlib import Path
import cv2
VALID_LABELS = [chr(code) for code in range(ord('A'), ord('Z') + 1)]
def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect webcam images for a specific sign label.',
    )
    parser.add_argument('--label', required=True, help='Letter label to capture, e.g. A')
    parser.add_argument(
        '--output-root',
        default='data\\custom_data',
        help='Root folder where captured images will be stored.',
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=50,
        help='How many images to save for the label.',
    )
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Camera index to open.',
    )
    parser.add_argument(
        '--countdown-seconds',
        type=float,
        default=3.0,
        help='Countdown before automatic capture starts.',
    )
    parser.add_argument(
        '--capture-interval',
        type=float,
        default=0.8,
        help='Seconds between automatic captures.',
    )
    return parser.parse_args()
def next_image_index(label_dir: Path) -> int:
    existing = []
    for path in label_dir.glob('*.jpg'):
        try:
            existing.append(int(path.stem))
        except ValueError:
            continue
    return max(existing, default=-1) + 1
def draw_overlay(frame, label, saved_count, target_count, status_lines):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (430, 165), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    base_lines = [
        f'Capture label: {label}',
        f'Saved: {saved_count}/{target_count}',
        'Press ESC to quit',
    ]
    lines = base_lines + status_lines
    y = 35
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return frame
def main():
    args = parse_args()
    label = args.label.upper().strip()
    if label not in VALID_LABELS:
        raise ValueError(f'Label must be A-Z. Got: {label}')
    output_root = Path(args.output_root)
    label_dir = output_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError('Unable to open webcam.')
    image_index = next_image_index(label_dir)
    saved_count = len(list(label_dir.glob('*.jpg')))
    capture_start = time.time() + args.countdown_seconds
    next_capture_time = capture_start
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError('Unable to read frame from webcam.')
            frame = cv2.flip(frame, 1)
            now = time.time()
            if now < capture_start:
                remaining = max(0.0, capture_start - now)
                status_lines = [
                    f'Get into position. Auto capture starts in {remaining:.1f}s',
                    'Hold the gesture steady when capture begins',
                ]
            else:
                remaining_to_capture = max(0.0, next_capture_time - now)
                status_lines = [
                    'Auto capture is running',
                    f'Next photo in {remaining_to_capture:.1f}s',
                ]
                if now >= next_capture_time and saved_count < args.target_count:
                    output_path = label_dir / f'{image_index:04d}.jpg'
                    cv2.imwrite(str(output_path), frame)
                    print(f'Saved: {output_path}')
                    image_index += 1
                    saved_count += 1
                    next_capture_time = now + args.capture_interval
                if saved_count >= args.target_count:
                    break
            display = draw_overlay(frame.copy(), label, saved_count, args.target_count, status_lines)
            cv2.imshow('Collect Webcam Samples', display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f'Finished label {label}: {saved_count} images available in {label_dir}')
if __name__ == '__main__':
    main()
