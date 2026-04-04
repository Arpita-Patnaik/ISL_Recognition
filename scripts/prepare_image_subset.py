import argparse
import shutil
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASSES = [chr(letter_code) for letter_code in range(ord("A"), ord("Z") + 1)]
def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy a fixed number of images per class into a clean dataset folder.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Source dataset root containing one folder per alphabet class.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "data_set"),
        help="Destination dataset root where sampled images will be copied.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=200,
        help="Maximum number of raw images to keep per class.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_CLASSES,
        help="Classes to include. Default is A-Z.",
    )
    return parser.parse_args()
def list_images(class_dir: Path):
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES
    )
def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    classes = [class_name.upper() for class_name in args.classes]
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        source_class_dir = input_dir / class_name
        if not source_class_dir.exists():
            print(f"Skipping missing class folder: {source_class_dir}")
            continue
        destination_class_dir = output_dir / class_name
        if destination_class_dir.exists():
            shutil.rmtree(destination_class_dir)
        destination_class_dir.mkdir(parents=True, exist_ok=True)
        images = list_images(source_class_dir)
        selected_images = images[: args.images_per_class]
        for index, image_path in enumerate(selected_images):
            destination_name = f"{index:04d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, destination_class_dir / destination_name)
        print(
            f"{class_name}: copied {len(selected_images)} images "
            f"(available: {len(images)})",
        )
if __name__ == "__main__":
    main()
