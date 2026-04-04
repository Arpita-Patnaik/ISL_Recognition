import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DEFAULT_CLASSES = [chr(letter_code) for letter_code in range(ord("A"), ord("Z") + 1)]
def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep an exact number of landmark rows per class.",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Source landmarks CSV.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Destination balanced CSV.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=150,
        help="Target number of valid rows to keep for each class.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_CLASSES,
        help="Classes to include. Default is A-Z.",
    )
    return parser.parse_args()
def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    classes = [class_name.upper() for class_name in args.classes]
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    rows_by_class = defaultdict(list)
    with input_csv.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        header = reader.fieldnames
        if not header or "label" not in header:
            raise ValueError("Input CSV must contain a 'label' column.")
        for row in reader:
            label = row["label"].upper()
            if label in classes:
                rows_by_class[label].append(row)
    missing_or_small = {
        class_name: len(rows_by_class[class_name])
        for class_name in classes
        if len(rows_by_class[class_name]) < args.samples_per_class
    }
    if missing_or_small:
        raise ValueError(
            "Not enough valid rows for some classes: "
            f"{missing_or_small}. Collect/extract more data first.",
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=header)
        writer.writeheader()
        for class_name in classes:
            selected_rows = rows_by_class[class_name][: args.samples_per_class]
            writer.writerows(selected_rows)
            print(f"{class_name}: kept {len(selected_rows)} rows")
    print(f"Saved balanced CSV: {output_csv}")
if __name__ == "__main__":
    main()
