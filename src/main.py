import cv2
from collections import Counter, deque
from src.config import (
    BUFFER_SIZE,
    DETECTION_CONFIDENCE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_HANDS,
    MEDIAPIPE_MODEL_PATH,
    MODEL_PATH,
    TRACKING_CONFIDENCE,
)
from src.model_loader import load_model, predict_letter
from src.ui_overlay import apply_full_overlay
from src.utils import (
    detect_hand,
    extract_landmarks,
    hand_detected,
    init_mediapipe,
    normalize_landmarks,
)
class ISLRecognitionApp:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.detector = init_mediapipe(
            model_path=MEDIAPIPE_MODEL_PATH,
            max_hands=MAX_HANDS,
            detection_conf=DETECTION_CONFIDENCE,
            tracking_conf=TRACKING_CONFIDENCE,
        )
        self.buffer = deque(maxlen=BUFFER_SIZE)
    def get_stable_prediction(self):
        if not self.buffer:
            return None
        return Counter(self.buffer).most_common(1)[0][0]
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        results, annotated = detect_hand(frame, self.detector)
        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        raw_letter = None
        confidence = 0.0
        if hand_detected(results):
            landmarks = extract_landmarks(results)
            normalized = normalize_landmarks(landmarks)
            raw_letter, confidence = predict_letter(self.model, normalized)
            if raw_letter:
                self.buffer.append(raw_letter)
        else:
            self.buffer.clear()
        stable_letter = self.get_stable_prediction()
        output_frame = apply_full_overlay(
            annotated,
            raw_letter=raw_letter,
            stable_letter=stable_letter,
            confidence=confidence,
            num_hands=num_hands,
        )
        return output_frame
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Unable to read frame from webcam.")
                output_frame = self.process_frame(frame)
                cv2.imshow("ISL Recognition", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
def main():
    app = ISLRecognitionApp()
    app.run()
if __name__ == "__main__":
    main()
