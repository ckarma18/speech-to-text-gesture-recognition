from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.pkl"
CONFIDENCE_THRESHOLD = 0.50

EMOJI_MAP = {
    "good": "👍",
    "bad": "👎",
    "stop": "✋",
    "go": "👉",
    "come": "👋",
    "ok": "👌",
    "call": "📞",
    "toilet": "🚻",
    "good_luck": "🤞",
    "silent": "🤫",
    "no_matching": "❓",
    "no_hand": "🖐️"
}

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class LandmarkData(BaseModel):
    landmarks: List[float]

def normalize_flat_landmarks(landmarks: List[float]) -> List[float]:
    if len(landmarks) != 42:
        return landmarks

    base_x = landmarks[0]
    base_y = landmarks[1]

    normalized = []
    for i in range(0, len(landmarks), 2):
        normalized.append(landmarks[i] - base_x)
        normalized.append(landmarks[i + 1] - base_y)

    return normalized

@app.get("/")
def root():
    return {"message": "Gesture API is running"}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict-gesture")
def predict(data: LandmarkData):
    if len(data.landmarks) != 42:
        return {
            "gesture": "no_hand",
            "emoji": EMOJI_MAP["no_hand"],
            "confidence": 0.0
        }

    processed_landmarks = normalize_flat_landmarks(data.landmarks)

    try:
        proba = model.predict_proba([processed_landmarks])[0]
        confidence = float(max(proba))
        prediction = model.classes_[proba.argmax()]
    except Exception:
        prediction = model.predict([processed_landmarks])[0]
        confidence = 1.0

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "gesture": "no_matching",
            "emoji": EMOJI_MAP["no_matching"],
            "confidence": round(confidence, 3)
        }

    return {
        "gesture": prediction,
        "emoji": EMOJI_MAP.get(prediction, ""),
        "confidence": round(confidence, 3)
    }