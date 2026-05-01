import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import csv
import msvcrt
import requests   # 🔥 for live testing

print("Script started...")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found.")
    exit()

os.makedirs("data", exist_ok=True)

allowed_labels = [
    "good", "bad", "stop", "go", "come",
    "ok", "call", "toilet", "good_luck", "silent"
]

label = input(f"Enter gesture label {allowed_labels}: ").strip().lower()

if label not in allowed_labels:
    print("Invalid label.")
    cap.release()
    exit()

file_path = "data/gesture_data.csv"
file = open(file_path, "a", newline="")
writer = csv.writer(file)

count = 0
target_samples = 50
last_row = []

# 🔥 NORMALIZATION (IMPORTANT)
def normalize_landmarks(hand_landmarks):
    coords = []

    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y

    for lm in hand_landmarks.landmark:
        coords.append(lm.x - base_x)
        coords.append(lm.y - base_y)

    # scale normalization
    max_value = max([abs(x) for x in coords]) or 1
    coords = [x / max_value for x in coords]

    return coords

print(f"\nCollecting data for: {label}")
print("Press S or SPACE to save sample")
print("Press ESC to quit")
print("Live prediction will show in terminal")
print(f"Target samples: {target_samples}\n")

while count < target_samples:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    last_row = []

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        last_row = normalize_landmarks(hand_landmarks)

        # 🔥 LIVE TEST (no frontend needed)
        if len(last_row) == 42:
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/api/predict-gesture",
                    json={"landmarks": last_row}
                )
                print(res.json())
            except:
                print("Backend not running")

        status_text = "Hand detected"
        color = (0, 255, 0)
    else:
        status_text = "No hand detected"
        color = (0, 0, 255)

    cv2.putText(frame, f"Gesture: {label}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Samples: {count}/{target_samples}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, "S/SPACE = save | ESC = quit", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, status_text, (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Gesture Collection + Live Test", frame)

    key = cv2.waitKey(1) & 0xFF

    terminal_key = None
    if msvcrt.kbhit():
        terminal_key = msvcrt.getwch()

    save_pressed = (
        key == ord("s") or key == ord("S") or key == 32 or
        terminal_key == "s" or terminal_key == "S" or terminal_key == " "
    )

    esc_pressed = (
        key == 27 or terminal_key == "\x1b"
    )

    if save_pressed:
        if len(last_row) == 42:
            saved_row = last_row.copy()
            saved_row.append(label)
            writer.writerow(saved_row)
            file.flush()
            count += 1
            print(f"✅ Saved sample {count}/{target_samples}")
        else:
            print("❌ No hand detected. Sample not saved.")

    if esc_pressed:
        print("ESC pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
file.close()
hands.close()

print(f"\nDone! Collected {count} samples for '{label}'")
print("Now run: python train_model.py")