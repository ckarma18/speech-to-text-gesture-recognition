import os
import pickle
import numpy as np
import pandas as pd
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "speech_dataset.csv"
MODEL_PATH = "speech_model.pkl"
LABEL_ENCODER_PATH = "speech_label_encoder.pkl"

def extract_features(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    features = np.concatenate([mfcc_mean, delta_mean])
    return features

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    if df.empty:
        print("❌ CSV is empty")
        return

    X = []
    y = []

    for _, row in df.iterrows():
        audio_path = row["audio_path"]
        text = str(row["text"]).strip()

        if not os.path.exists(audio_path):
            print(f"⚠️ Missing audio file: {audio_path}")
            continue

        try:
            features = extract_features(audio_path)
            X.append(features)
            y.append(text)
            print(f"✅ Processed: {audio_path} -> {text}")
        except Exception as e:
            print(f"⚠️ Failed: {audio_path} -> {e}")

    if len(X) < 2:
        print("❌ Not enough training samples")
        return

    X = np.array(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y_encoded)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print("\n✅ Custom speech model trained successfully!")
    print(f"✅ Saved model: {MODEL_PATH}")
    print(f"✅ Saved label encoder: {LABEL_ENCODER_PATH}")
    print(f"✅ Classes: {list(label_encoder.classes_)}")

if __name__ == "__main__":
    main()