import argparse
import shutil
from pathlib import Path
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASSES = [chr(letter_code) for letter_code in range(ord("A"), ord("Z") + 1)]
DEFAULT_WEAK_CLASSES = ["A", "G", "I", "M", "O", "P", "Q", "S", "V", "X", "Y", "Z"]
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a mixed dataset: cap strong classes, but keep all images for weak classes."
        ),
    )
    parser.add_argument("--input-dir", required=True, help="Source dataset root.")
    parser.add_argument("--output-dir", required=True, help="Destination dataset root.")
    parser.add_argument(
        "--default-images-per-class",
        type=int,
        default=200,
        help="Image cap for strong classes.",
    )
    parser.add_argument(
        "--weak-classes",
        nargs="*",
        default=DEFAULT_WEAK_CLASSES,
        help="Classes that should keep all available images.",
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
    weak_classes = {class_name.upper() for class_name in args.weak_classes}
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
        if class_name in weak_classes:
            selected_images = images
            mode = "all"
        else:
            selected_images = images[: args.default_images_per_class]
            mode = f"cap={args.default_images_per_class}"
        for index, image_path in enumerate(selected_images):
            destination_name = f"{index:05d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, destination_class_dir / destination_name)
        print(
            f"{class_name}: copied {len(selected_images)} images "
            f"(available: {len(images)}, mode: {mode})"
        )
if __name__ == "__main__":
    main()
