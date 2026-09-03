"""
Real-time finger counter using OpenCV + MediaPipe's HandLandmarker (Tasks API).

MediaPipe >= 1.0 removed the old `mp.solutions.hands` API in favor of the
Tasks API used here. On first run this script downloads a small hand
landmark model file (~8 MB) into the same folder and reuses it afterwards.

Install dependencies:
    pip install opencv-python mediapipe numpy

Run:
    python finger_counter.py

Press 'q' to quit.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Landmark indices for fingertips and their corresponding lower joints
# (thumb, index, middle, ring, pinky)
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def count_fingers(landmarks, handedness_label):
    """Return the number of extended fingers for one detected hand."""
    fingers_up = []

    # Thumb: compare x-coordinates since the thumb moves sideways, not up/down.
    # "Right" hand as seen in a mirrored (selfie) camera view is the user's
    # actual left hand, so flip the comparison based on handedness label.
    if handedness_label == "Right":
        fingers_up.append(landmarks[TIP_IDS[0]].x < landmarks[PIP_IDS[0]].x)
    else:
        fingers_up.append(landmarks[TIP_IDS[0]].x > landmarks[PIP_IDS[0]].x)

    # Other four fingers: tip above the lower joint (smaller y = higher on screen)
    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers_up.append(landmarks[tip_id].y < landmarks[pip_id].y)

    return sum(fingers_up)


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)


def main():
    ensure_model()

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_timestamp_ms = 0
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie view
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp_ms += 33  # assume ~30 fps
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            total_fingers = 0
            if result.hand_landmarks:
                for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name
                    total_fingers += count_fingers(hand_landmarks, label)
                    draw_landmarks(frame, hand_landmarks)

            cv2.putText(
                frame,
                f"Fingers: {total_fingers}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )

            cv2.imshow("Finger Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
