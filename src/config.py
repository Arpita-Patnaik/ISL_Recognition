# config.py
# Single place for all project settings.
# Change things here — everything else picks it up automatically.

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH           = os.path.join(BASE_DIR, "model", "isl_model.pkl")
MEDIAPIPE_MODEL_PATH = os.path.join(BASE_DIR, "model", "hand_landmarker.task")

# ── Camera ───────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0        # 0 = default webcam, change if using external camera

# ── MediaPipe ────────────────────────────────────────────────
MAX_HANDS            = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE  = 0.7

# ── Prediction ───────────────────────────────────────────────
BUFFER_SIZE          = 15    # frames to average over for smoothing
CONFIDENCE_THRESHOLD = 0.60  # minimum confidence to accept a prediction

# ── Speech ───────────────────────────────────────────────────
SPEECH_RATE   = 150     # words per minute
SPEECH_GENDER = "female"

# ── UI ───────────────────────────────────────────────────────
APP_TITLE      = "GestureVoice"
WINDOW_WIDTH   = 1100
WINDOW_HEIGHT  = 720
BG_COLOR       = "#d6d3e0"   # gray-purple background matching screenshot
PANEL_COLOR    = "#ffffff"   # white card panels
BUTTON_COLOR   = "#8b7fd4"   # purple buttons
BUTTON_TEXT    = "#ffffff"