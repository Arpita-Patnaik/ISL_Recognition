import cv2
def draw_prediction(frame, stable_letter, confidence):
    if stable_letter is None:
        return frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (160, 92), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame,
        str(stable_letter),
        (30, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.3,
        (0, 255, 136),
        4,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (10, 98), (160, 114), (50, 50, 50), -1)
    bar_width = int(150 * max(0.0, min(1.0, confidence)))
    cv2.rectangle(frame, (10, 98), (10 + bar_width, 114), (0, 200, 100), -1)
    conf_text = f"{int(confidence * 100)}%"
    cv2.putText(
        frame,
        conf_text,
        (60, 111),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame
def draw_no_hand(frame):
    cv2.putText(
        frame,
        "No hand detected",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (60, 60, 200),
        2,
        cv2.LINE_AA,
    )
    return frame
def draw_low_confidence(frame, confidence):
    cv2.putText(
        frame,
        f"Low confidence: {int(confidence * 100)}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
def draw_hand_count(frame, num_hands):
    text = f"Hands: {num_hands}"
    color = (0, 255, 0) if num_hands > 0 else (100, 100, 100)
    cv2.putText(
        frame,
        text,
        (frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame
def draw_title(frame):
    h = frame.shape[0]
    cv2.putText(
        frame,
        "ISL Recognition Debug",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return frame
def draw_debug_panel(frame, raw_letter, stable_letter, confidence):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 125), (250, 205), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    raw_text = raw_letter if raw_letter is not None else "None"
    stable_text = stable_letter if stable_letter is not None else "None"
    lines = [
        f"Raw: {raw_text}",
        f"Stable: {stable_text}",
        f"Confidence: {confidence:.2f}",
    ]
    y = 150
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return frame
def apply_full_overlay(frame, raw_letter, stable_letter, confidence, num_hands):
    frame = draw_hand_count(frame, num_hands)
    frame = draw_title(frame)
    frame = draw_debug_panel(frame, raw_letter, stable_letter, confidence)
    if num_hands == 0:
        frame = draw_no_hand(frame)
    elif stable_letter is None:
        frame = draw_low_confidence(frame, confidence)
    else:
        frame = draw_prediction(frame, stable_letter, confidence)
    return frame
