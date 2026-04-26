import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "data/gesture_data.csv"
MODEL_PATH = "model.pkl"

if not os.path.exists(DATASET_PATH):
    print(f"Dataset not found: {DATASET_PATH}")
    exit()

data = pd.read_csv(DATASET_PATH, header=None)

if data.shape[1] != 43:
    print(f"Unexpected dataset shape: {data.shape}")
    print("Expected 43 columns: 42 landmarks + 1 label")
    exit()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

if len(y.unique()) < 2:
    print("Need at least 2 different gesture classes to train the model.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved successfully as: {MODEL_PATH}")