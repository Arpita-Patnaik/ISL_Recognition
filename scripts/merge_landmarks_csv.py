import argparse
import csv
from pathlib import Path
def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge a base landmarks CSV with an extra landmarks CSV.',
    )
    parser.add_argument('--base-csv', required=True, help='Base balanced CSV path.')
    parser.add_argument('--extra-csv', required=True, help='Extra calibration CSV path.')
    parser.add_argument('--output-csv', required=True, help='Merged CSV output path.')
    parser.add_argument(
        '--repeat-extra',
        type=int,
        default=3,
        help='How many times to repeat extra rows to give them more influence.',
    )
    return parser.parse_args()
def read_rows(path: Path):
    with path.open('r', newline='', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        header = reader.fieldnames
        if not header or 'label' not in header:
            raise ValueError(f"CSV must contain a 'label' column: {path}")
        rows = list(reader)
    return header, rows
def main():
    args = parse_args()
    base_csv = Path(args.base_csv)
    extra_csv = Path(args.extra_csv)
    output_csv = Path(args.output_csv)
    if not base_csv.exists():
        raise FileNotFoundError(f'Base CSV not found: {base_csv}')
    if not extra_csv.exists():
        raise FileNotFoundError(f'Extra CSV not found: {extra_csv}')
    if args.repeat_extra < 1:
        raise ValueError('--repeat-extra must be at least 1')
    base_header, base_rows = read_rows(base_csv)
    extra_header, extra_rows = read_rows(extra_csv)
    if base_header != extra_header:
        raise ValueError('Base CSV and extra CSV have different columns.')
    merged_rows = list(base_rows)
    for _ in range(args.repeat_extra):
        merged_rows.extend(extra_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=base_header)
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f'Base rows: {len(base_rows)}')
    print(f'Extra rows: {len(extra_rows)}')
    print(f'Repeat extra: {args.repeat_extra}')
    print(f'Total merged rows: {len(merged_rows)}')
    print(f'Saved merged CSV: {output_csv}')
if __name__ == '__main__':
    main()
