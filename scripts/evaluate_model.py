import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


EXPECTED_FEATURES = 126


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ISL model against a landmarks CSV file.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(ROOT_DIR / "data" / "data" / "landmarks_data_2hand.csv"),
        help="Path to the landmarks CSV file.",
    )
    parser.add_argument(
        "--model-path",
        default=str(ROOT_DIR / "model" / "isl_model.pkl"),
        help="Path to the trained model pickle file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    model_path = Path(args.model_path)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    df = pd.read_csv(input_csv)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    feature_columns = [column for column in df.columns if column != "label"]
    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} feature columns, found {len(feature_columns)}.",
        )

    with model_path.open("rb") as file_obj:
        model = pickle.load(file_obj)

    X = df[feature_columns]
    y = df["label"]

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"Rows evaluated: {len(df)}")
    print(f"Accuracy on provided CSV: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=sorted(y.unique())))


if __name__ == "__main__":
    main()
