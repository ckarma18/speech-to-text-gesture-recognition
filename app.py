"""
Unified FastAPI Backend
- Speech-to-Text using Pretrained Whisper
- Speech-to-Text using Custom Trained KNN + MFCC model
- Gesture Recognition using Custom Trained KNN model
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import shutil
import time
import pickle
import os
from pathlib import Path

import librosa
import numpy as np

# =============================================================================
# LOAD MODELS
# =============================================================================

print("🔧 Loading trained KNN gesture model...")
try:
    with open("model.pkl", "rb") as f:
        gesture_model = pickle.load(f)
    GESTURE_MODEL_AVAILABLE = True
    print("✅ Trained KNN gesture model loaded")
except FileNotFoundError:
    gesture_model = None
    GESTURE_MODEL_AVAILABLE = False
    print("⚠️ model.pkl not found — run train_model.py first")

print("🔧 Loading custom trained speech model...")
try:
    with open("speech_model.pkl", "rb") as f:
        custom_speech_model = pickle.load(f)

    with open("speech_label_encoder.pkl", "rb") as f:
        speech_label_encoder = pickle.load(f)

    CUSTOM_SPEECH_AVAILABLE = True
    print("✅ Custom trained speech model loaded")
except FileNotFoundError:
    custom_speech_model = None
    speech_label_encoder = None
    CUSTOM_SPEECH_AVAILABLE = False
    print("⚠️ speech_model.pkl or speech_label_encoder.pkl not found")

print("🔧 Loading Speech-to-Text system (Whisper)...")
try:
    from main import SpeechToTextSystem
    speech_system = SpeechToTextSystem()
    SPEECH_AVAILABLE = True
    print("✅ Speech system ready")
except Exception as e:
    speech_system = None
    SPEECH_AVAILABLE = False
    print(f"⚠️ Speech system unavailable: {e}")

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Speech-to-Text + Gesture Recognition API",
    description="Unified API: Whisper ASR + Custom Speech Model + Gesture Model",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("api_results")
RESULTS_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.50

EMOJI_MAP = {
    "yes": "👍",
    "no": "👎",
    "peace": "✌️",
    "help": "🚨",
    "call": "📞",
    "come": "👋",
    "stop": "✋",
    "ok": "👌",
    "police": "🚓",
    "no_matching": "❓",
    "no_hand": "🖐️",
}

# =============================================================================
# DATA MODELS
# =============================================================================

class LandmarkData(BaseModel):
    landmarks: List[float]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_flat_landmarks(landmarks: List[float]) -> List[float]:
    """Normalize flat [x0,y0,x1,y1,...] landmarks relative to wrist."""
    if len(landmarks) != 42:
        return landmarks

    base_x = landmarks[0]
    base_y = landmarks[1]

    normalized = []
    for i in range(0, len(landmarks), 2):
        normalized.append(landmarks[i] - base_x)
        normalized.append(landmarks[i + 1] - base_y)

    return normalized


def extract_speech_features(audio_path: str) -> np.ndarray:
    """Extract MFCC + Delta features for custom speech model."""
    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    features = np.concatenate([mfcc_mean, delta_mean])
    return features


def validate_audio_extension(filename: str):
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    file_ext = Path(filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
        )

# =============================================================================
# ROOT & HEALTH
# =============================================================================

@app.get("/")
async def root():
    return {
        "name": "Speech-to-Text + Gesture Recognition API",
        "version": "4.0.0",
        "speech_available": SPEECH_AVAILABLE,
        "custom_speech_available": CUSTOM_SPEECH_AVAILABLE,
        "gesture_model_available": GESTURE_MODEL_AVAILABLE,
        "endpoints": {
            "health": "GET /api/health",
            "transcribe": "POST /api/transcribe",
            "transcribe_custom": "POST /api/transcribe-custom",
            "predict_gesture": "POST /api/predict-gesture",
            "models": "GET /api/models",
            "gestures": "GET /api/gestures",
            "stats": "GET /api/stats",
            "docs": "/docs"
        }
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Speech-to-Text + Gesture Recognition API",
        "version": "4.0.0",
        "speech_available": SPEECH_AVAILABLE,
        "custom_speech_available": CUSTOM_SPEECH_AVAILABLE,
        "gesture_model_available": GESTURE_MODEL_AVAILABLE,
    }

# =============================================================================
# SPEECH - PRETRAINED WHISPER
# =============================================================================

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe a single audio file using pretrained Whisper."""
    if not SPEECH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Pretrained Whisper speech system not available."
        )

    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    validate_audio_extension(audio.filename)

    file_path = UPLOAD_DIR / audio.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        start_time = time.time()
        result = speech_system.transcribe_audio(str(file_path))
        processing_time = time.time() - start_time

        response = {
            "transcription": result.get("transcription", ""),
            "confidence": result.get("confidence", 0.0),
            "processing_time": processing_time,
            "language": result.get("language", "en"),
            "model": result.get("model", "openai/whisper-base"),
            "model_type": "pretrained",
            "input_mode": "upload",
            "file": audio.filename,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        if file_path.exists():
            file_path.unlink()

# =============================================================================
# SPEECH - CUSTOM TRAINED MODEL
# =============================================================================

@app.post("/api/transcribe-custom")
async def transcribe_custom(audio: UploadFile = File(...)):
    """Transcribe using custom trained KNN + MFCC speech model."""
    if not CUSTOM_SPEECH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Custom trained speech model not available. Run train_simple_asr.py first."
        )

    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    validate_audio_extension(audio.filename)

    file_path = UPLOAD_DIR / audio.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        start_time = time.time()

        features = extract_speech_features(str(file_path))
        pred_encoded = custom_speech_model.predict([features])[0]
        prediction = speech_label_encoder.inverse_transform([pred_encoded])[0]

        processing_time = time.time() - start_time

        response = {
            "transcription": prediction,
            "confidence": 1.0,
            "processing_time": processing_time,
            "language": "en",
            "model": "Custom KNN Speech Model",
            "model_type": "custom",
            "input_mode": "upload",
            "file": audio.filename,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom transcription error: {str(e)}")
    finally:
        if file_path.exists():
            file_path.unlink()

# =============================================================================
# GESTURE ENDPOINTS
# =============================================================================

@app.get("/api/gestures")
async def list_gestures():
    trained_gestures = []
    if GESTURE_MODEL_AVAILABLE:
        try:
            for label in gesture_model.classes_:
                trained_gestures.append({
                    "name": label.upper(),
                    "raw_label": label,
                    "emoji": EMOJI_MAP.get(label, "✋"),
                    "description": f"Trained gesture: {label}",
                    "source": "custom_trained_model",
                })
        except Exception:
            pass

    return JSONResponse(content={
        "trained_gestures": trained_gestures,
        "gesture_model_ready": GESTURE_MODEL_AVAILABLE,
        "total_trained": len(trained_gestures),
    }, status_code=200)


@app.post("/api/predict-gesture")
async def predict_gesture(data: LandmarkData):
    """Predict gesture using your trained KNN gesture model."""
    if not GESTURE_MODEL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Trained gesture model not available. Run train_model.py to create model.pkl"
        )

    if len(data.landmarks) != 42:
        return JSONResponse(content={
            "gesture": "no_hand",
            "emoji": EMOJI_MAP["no_hand"],
            "confidence": 0.0
        }, status_code=200)

    try:
        processed_landmarks = normalize_flat_landmarks(data.landmarks)

        try:
            proba = gesture_model.predict_proba([processed_landmarks])[0]
            confidence = float(max(proba))
            prediction = gesture_model.classes_[proba.argmax()]
        except Exception:
            prediction = gesture_model.predict([processed_landmarks])[0]
            confidence = 1.0

        if confidence < CONFIDENCE_THRESHOLD:
            return JSONResponse(content={
                "gesture": "no_matching",
                "emoji": EMOJI_MAP["no_matching"],
                "confidence": round(confidence, 3)
            }, status_code=200)

        return JSONResponse(content={
            "gesture": prediction,
            "emoji": EMOJI_MAP.get(prediction, ""),
            "confidence": round(confidence, 3),
            "model": "Custom KNN Gesture Model",
            "model_type": "custom"
        }, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gesture prediction error: {str(e)}")

# =============================================================================
# MODELS INFO
# =============================================================================

@app.get("/api/models")
async def list_models():
    return JSONResponse(content={
        "speech_models": [
            {
                "name": "openai/whisper-base",
                "type": "pretrained",
                "status": "available" if SPEECH_AVAILABLE else "unavailable",
                "description": "Pretrained Whisper model for speech transcription"
            },
            {
                "name": "Custom KNN Speech Model",
                "type": "custom_trained",
                "status": "available" if CUSTOM_SPEECH_AVAILABLE else "unavailable",
                "description": "Custom trained speech model using MFCC + KNN"
            }
        ],
        "gesture_models": [
            {
                "name": "Custom KNN Gesture Model",
                "type": "custom_trained",
                "status": "available" if GESTURE_MODEL_AVAILABLE else "unavailable",
                "description": "Custom trained gesture model using MediaPipe landmarks + KNN"
            }
        ]
    }, status_code=200)

# =============================================================================
# EVALUATE
# =============================================================================

@app.post("/api/evaluate")
async def evaluate_transcription(data: dict):
    try:
        hypothesis = data.get("hypothesis", "")
        reference = data.get("reference", "")

        if not hypothesis or not reference:
            raise HTTPException(
                status_code=400,
                detail="Both hypothesis and reference text required"
            )

        from evaluation import Evaluator
        wer_result, _ = Evaluator.calculate_wer(hypothesis, reference)
        cer_result, _ = Evaluator.calculate_cer(hypothesis, reference)

        return JSONResponse(content={
            "wer": wer_result,
            "cer": cer_result,
            "accuracy": round(1 - wer_result / 100, 3)
        }, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# STATS
# =============================================================================

@app.get("/api/stats")
async def get_statistics():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    except Exception:
        gpu_available, gpu_count, gpu_name = False, 0, "N/A"

    gesture_labels = []
    if GESTURE_MODEL_AVAILABLE:
        try:
            gesture_labels = list(gesture_model.classes_)
        except Exception:
            pass

    speech_labels = []
    if CUSTOM_SPEECH_AVAILABLE:
        try:
            speech_labels = list(speech_label_encoder.classes_)
        except Exception:
            pass

    return JSONResponse(content={
        "system": {
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
        },
        "speech": {
            "pretrained_available": SPEECH_AVAILABLE,
            "pretrained_model": "openai/whisper-base",
        },
        "custom_speech": {
            "model_available": CUSTOM_SPEECH_AVAILABLE,
            "model_type": "KNN + MFCC",
            "trained_classes": speech_labels,
            "num_classes": len(speech_labels),
        },
        "gesture": {
            "model_available": GESTURE_MODEL_AVAILABLE,
            "model_type": "KNN + MediaPipe landmarks",
            "trained_gestures": gesture_labels,
            "num_gestures": len(gesture_labels),
        },
        "upload_directory": str(UPLOAD_DIR),
        "results_directory": str(RESULTS_DIR),
    }, status_code=200)

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Unified Speech + Gesture API...")
    print("📖 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")