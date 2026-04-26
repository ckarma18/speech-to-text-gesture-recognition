import cv2
import mediapipe as mp
import pickle
from collections import deque

print("Prediction script started...")

MODEL_PATH = "model.pkl"
CONFIDENCE_THRESHOLD = 0.45

EMOJI_MAP = {
    "help": "🚨",
    "yes": "👍",
    "no": "👎",
    "call": "📞",
    "come": "👋",
    "stop": "✋",
    "ok": "👌",
    "police": "🚓",
    "peace": "☮️",
    "no_matching": "❓"
}

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

gesture_history = deque(maxlen=6)

def normalize_landmarks(hand_landmarks):
    coords = []
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y

    for lm in hand_landmarks.landmark:
        coords.append(lm.x - base_x)
        coords.append(lm.y - base_y)

    return coords

def is_help_stable(prediction: str) -> bool:
    gesture_history.append(prediction)
    return len(gesture_history) >= 4 and all(g == "help" for g in list(gesture_history)[-4:])

if not cap.isOpened():
    print("Camera not working")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display_text = "No hand detected"
    display_color = (0, 0, 255)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        row = normalize_landmarks(hand_landmarks)

        if len(row) == 42:
            try:
                proba = model.predict_proba([row])[0]
                confidence = float(max(proba))
                prediction = model.classes_[proba.argmax()]
            except Exception:
                prediction = model.predict([row])[0]
                confidence = 1.0

            if confidence < CONFIDENCE_THRESHOLD:
                prediction = "no_matching"
                display_text = f"NO MATCHING ❓ ({round(confidence, 3)})"
                display_color = (0, 255, 255)
            else:
                help_detected = is_help_stable(prediction)
                emoji = EMOJI_MAP.get(prediction, "")

                if help_detected:
                    display_text = f"HELP DETECTED! 🚨 ({round(confidence, 3)})"
                    display_color = (0, 0, 255)
                else:
                    display_text = f"{prediction.upper()} {emoji} ({round(confidence, 3)})"
                    display_color = (0, 255, 0)

    cv2.putText(frame, display_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)

    cv2.imshow("Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()