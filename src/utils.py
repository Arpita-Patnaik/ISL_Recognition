# utils.py
# MediaPipe hand detection — 2-hand support (126 features)
# Clean, consistent, production-ready version

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =========================
# CONSTANTS
# =========================

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17)
]

LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 126

# =========================
# MODEL DOWNLOAD
# =========================

def download_model(save_path):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    print("⬇️ Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(url, save_path)
    print(f"✅ Model saved to: {save_path}")

# =========================
# INIT MEDIAPIPE
# =========================

def init_mediapipe(model_path, max_hands=2, detection_conf=0.7, tracking_conf=0.7):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=max_hands,
        min_hand_detection_confidence=detection_conf,
        min_hand_presence_confidence=tracking_conf,
        running_mode=mp_vision.RunningMode.IMAGE
    )

    return mp_vision.HandLandmarker.create_from_options(options)

# =========================
# DETECTION
# =========================

def detect_hand(frame, detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    annotated = frame.copy()
    h, w = frame.shape[:2]

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for s, e in HAND_CONNECTIONS:
                cv2.line(annotated, points[s], points[e], (0, 200, 0), 2)

            for (px, py) in points:
                cv2.circle(annotated, (px, py), 5, (0, 0, 255), -1)

    return results, annotated

# =========================
# HAND CHECK
# =========================

def hand_detected(results):
    return results.hand_landmarks is not None and len(results.hand_landmarks) > 0

# =========================
# LANDMARK EXTRACTION (FIXED ORDER)
# =========================

def extract_landmarks(results):
    """
    Returns 126 features:
    [Left hand (63), Right hand (63)]

    Missing hand → zero padded
    """

    if not hand_detected(results):
        return None

    # Initialize with zeros
    left_hand = [0.0] * FEATURES_PER_HAND
    right_hand = [0.0] * FEATURES_PER_HAND

    # Safety check
    if results.handedness is None:
        return None

    if len(results.hand_landmarks) != len(results.handedness):
        return None

    for hand, handedness in zip(results.hand_landmarks, results.handedness):

        label = handedness[0].category_name  # "Left" or "Right"

        features = []
        for lm in hand:
            features.extend([lm.x, lm.y, lm.z])

        if len(features) != FEATURES_PER_HAND:
            continue

        if label == "Left":
            left_hand = features
        elif label == "Right":
            right_hand = features

    return left_hand + right_hand  # always 126

# =========================
# NORMALIZATION
# =========================

def normalize_landmarks(landmarks):
    """
    Normalize relative to wrist (landmark 0 of each hand)
    """

    if landmarks is None or len(landmarks) != TOTAL_FEATURES:
        return None

    # Hand 1 (Left)
    wx1, wy1, wz1 = landmarks[0], landmarks[1], landmarks[2]
    hand1_norm = []

    for i in range(0, FEATURES_PER_HAND, 3):
        hand1_norm.extend([
            landmarks[i] - wx1,
            landmarks[i+1] - wy1,
            landmarks[i+2] - wz1
        ])

    # Hand 2 (Right)
    hand2_raw = landmarks[FEATURES_PER_HAND:]
    hand2_norm = []

    # Check if hand 2 exists
    if any(v != 0.0 for v in hand2_raw):
        wx2, wy2, wz2 = hand2_raw[0], hand2_raw[1], hand2_raw[2]

        for i in range(0, FEATURES_PER_HAND, 3):
            hand2_norm.extend([
                hand2_raw[i] - wx2,
                hand2_raw[i+1] - wy2,
                hand2_raw[i+2] - wz2
            ])
    else:
        hand2_norm = [0.0] * FEATURES_PER_HAND

    return hand1_norm + hand2_norm