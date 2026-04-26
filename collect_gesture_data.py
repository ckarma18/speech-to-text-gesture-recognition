import cv2
import mediapipe as mp
import csv
import os

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

os.makedirs("data", exist_ok=True)

allowed_labels = ["yes", "no", "peace"]

label = input(f"Enter gesture label {allowed_labels}: ").strip().lower()

if label not in allowed_labels:
    print("Invalid label. Please choose from the list.")
    cap.release()
    exit()

file_path = "data/gesture_data.csv"
file = open(file_path, "a", newline="")
writer = csv.writer(file)

count = 0
target_samples = 50

def normalize_landmarks(hand_landmarks):
    coords = []
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y

    for lm in hand_landmarks.landmark:
        coords.append(lm.x - base_x)
        coords.append(lm.y - base_y)

    return coords

print(f"Collecting data for: {label}")
print("Press 'S' to save sample, ESC to exit")
print(f"Aim for {target_samples} samples minimum — vary hand position and distance!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read from camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    row = []

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        row = normalize_landmarks(hand_landmarks)

    cv2.putText(frame, f"Gesture: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Samples: {count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Press S to save | ESC to quit", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if result.multi_hand_landmarks:
        cv2.putText(frame, "Hand detected!", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Gesture Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(row) == 42:
        row.append(label)
        writer.writerow(row)
        file.flush()
        count += 1
        print(f"Saved sample {count} for '{label}'")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
file.close()

print(f"\nDone! Collected {count} samples for '{label}'")
print("Now run: python train_model.py")