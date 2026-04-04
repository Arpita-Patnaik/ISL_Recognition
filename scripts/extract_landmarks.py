import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DETECTION_CONFIDENCE,
    MAX_HANDS,
    MEDIAPIPE_MODEL_PATH,
    TRACKING_CONFIDENCE,
)
from src.utils import detect_hand, extract_landmarks, init_mediapipe, normalize_landmarks


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract normalized hand landmarks from an image dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT_DIR / "data" / "data_set"),
        help="Directory containing one subfolder per letter/class.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT_DIR / "data" / "data" / "landmarks_data_2hand.csv"),
        help="Output CSV path for extracted features.",
    )
    return parser.parse_args()


def iter_class_images(dataset_dir: Path):
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                yield class_dir.name, image_path


def write_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [f"f{i}" for i in range(126)] + ["label"]
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_csv = Path(args.output_csv)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    detector = init_mediapipe(
        model_path=MEDIAPIPE_MODEL_PATH,
        max_hands=MAX_HANDS,
        detection_conf=DETECTION_CONFIDENCE,
        tracking_conf=TRACKING_CONFIDENCE,
    )

    rows = []
    total_images = 0
    skipped_images = 0
    saved_per_class = {}
    skipped_per_class = {}

    for label, image_path in iter_class_images(dataset_dir):
        total_images += 1
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_images += 1
            skipped_per_class[label] = skipped_per_class.get(label, 0) + 1
            continue

        results, _ = detect_hand(image, detector)
        landmarks = extract_landmarks(results)
        normalized = normalize_landmarks(landmarks)

        if normalized is None:
            skipped_images += 1
            skipped_per_class[label] = skipped_per_class.get(label, 0) + 1
            continue

        rows.append(normalized + [label])
        saved_per_class[label] = saved_per_class.get(label, 0) + 1

    write_csv(rows, output_csv)

    print(f"Saved CSV: {output_csv}")
    print(f"Total images scanned: {total_images}")
    print(f"Rows written: {len(rows)}")
    print(f"Rows skipped: {skipped_images}")

    if saved_per_class:
        print("Saved per class:")
        for label in sorted(saved_per_class):
            print(f"  {label}: {saved_per_class[label]}")

    if skipped_per_class:
        print("Skipped per class:")
        for label in sorted(skipped_per_class):
            print(f"  {label}: {skipped_per_class[label]}")


if __name__ == "__main__":
    main()
