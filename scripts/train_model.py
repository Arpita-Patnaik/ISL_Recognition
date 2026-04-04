import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


EXPECTED_FEATURES = 126


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an ISL classifier from extracted landmarks.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(ROOT_DIR / "data" / "data" / "landmarks_data_2hand.csv"),
        help="Path to the landmarks CSV file.",
    )
    parser.add_argument(
        "--output-model",
        default=str(ROOT_DIR / "model" / "isl_model.pkl"),
        help="Path where the trained model will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and training.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the random forest.",
    )
    return parser.parse_args()


def validate_dataframe(df: pd.DataFrame):
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    feature_columns = [column for column in df.columns if column != "label"]
    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} feature columns, found {len(feature_columns)}.",
        )

    if df.isna().sum().sum() > 0:
        raise ValueError("Input CSV contains missing values.")

    class_counts = df["label"].value_counts().sort_index()
    if (class_counts < 2).any():
        too_small = class_counts[class_counts < 2]
        raise ValueError(
            f"Each class needs at least 2 samples. Too small: {too_small.to_dict()}",
        )

    return feature_columns, class_counts


def save_metadata(output_model: Path, accuracy: float, class_counts):
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier",
        "expected_features": EXPECTED_FEATURES,
        "accuracy": accuracy,
        "class_counts": class_counts.to_dict(),
        "sklearn_version": sklearn.__version__,
    }

    metadata_path = output_model.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_model = Path(args.output_model)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    feature_columns, class_counts = validate_dataframe(df)

    X = df[feature_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, predictions))

    output_model.parent.mkdir(parents=True, exist_ok=True)
    with output_model.open("wb") as file_obj:
        pickle.dump(model, file_obj)
    print(f"Saved model: {output_model}")

    save_metadata(output_model, accuracy, class_counts)


if __name__ == "__main__":
    main()
