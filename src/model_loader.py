# model_loader.py
# Updated for 2-hand support.
# Model now expects 126 features instead of 63.

import pickle
import os
import numpy as np

CONFIDENCE_THRESHOLD = 0.6


def load_model(model_path):
    """Loads the trained model. Now expects 126 features."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded!")
    print(f"  Letters  : {list(model.classes_)}")
    print(f"  Features : {model.n_features_in_} (expected: 126 for 2-hand)")

    # Warn if model still expects 63 — means it needs retraining
    if model.n_features_in_ == 63:
        print("  WARNING: Model expects 63 features (single hand).")
        print("  Ask ML engineer to retrain with 2-hand data (126 features).")
    elif model.n_features_in_ == 126:
        print("  2-hand model confirmed!")

    return model


def predict_letter(model, landmarks):
    """
    Predicts ISL letter from 126 landmark features.
    Returns (letter, confidence) or (None, confidence) if below threshold.
    """
    if landmarks is None:
        return None, 0.0

    input_data = np.array(landmarks).reshape(1, -1)

    # Safety check — if feature count mismatches, warn clearly
    if input_data.shape[1] != model.n_features_in_:
        print(f"  Feature mismatch! Got {input_data.shape[1]}, "
              f"model expects {model.n_features_in_}")
        return None, 0.0

    probabilities    = model.predict_proba(input_data)[0]
    confidence       = float(max(probabilities))
    predicted_idx    = int(np.argmax(probabilities))
    predicted_letter = model.classes_[predicted_idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        return predicted_letter, confidence
    else:
        return None, confidence
