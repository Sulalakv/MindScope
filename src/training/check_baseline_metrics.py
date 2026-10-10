from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

VAL_CSV = ROOT / "dataset_val/raw/PHQ/val.csv"

VAL_FACE = ROOT / "dataset_val/features/face_agg"
VAL_AUDIO = ROOT / "dataset_val/features/audio_mfcc"

MODEL_DIR = ROOT / "models/baseline"

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class BaselineFusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1280 + 40, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 2)
        )

    def forward(self, face, audio):

        combined = torch.cat(
            [face, audio],
            dim=1
        )

        return self.network(combined)


# ============================================================
# LOAD MODEL
# ============================================================

model = BaselineFusionModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "best_model.pt",
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# LOAD TRAINING SCALERS
# ============================================================

face_scaler = joblib.load(
    MODEL_DIR / "face_scaler.pkl"
)

audio_scaler = joblib.load(
    MODEL_DIR / "audio_scaler.pkl"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

df = pd.read_csv(VAL_CSV)

subjects = []
faces = []
audios = []
labels = []


for _, row in df.iterrows():

    subject_id = str(row["subject_id"])
    label = int(row["Label"])

    face = np.load(
        VAL_FACE / f"{subject_id}.npy"
    )

    audio = np.load(
        VAL_AUDIO / f"{subject_id}.npy"
    )

    subjects.append(subject_id)
    faces.append(face)
    audios.append(audio)
    labels.append(label)


faces = np.asarray(
    faces,
    dtype=np.float32
)

audios = np.asarray(
    audios,
    dtype=np.float32
)

labels = np.asarray(
    labels,
    dtype=np.int64
)


# ============================================================
# APPLY TRAINING SCALERS
# ============================================================

faces = face_scaler.transform(faces)
audios = audio_scaler.transform(audios)


# ============================================================
# PREDICT
# ============================================================

face_tensor = torch.tensor(
    faces,
    dtype=torch.float32
)

audio_tensor = torch.tensor(
    audios,
    dtype=torch.float32
)


with torch.no_grad():

    outputs = model(
        face_tensor,
        audio_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predictions = torch.argmax(
        outputs,
        dim=1
    ).numpy()


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    labels,
    predictions
)

precision = precision_score(
    labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    labels,
    predictions
)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("INDEPENDENT BASELINE METRIC CHECK")
print("=" * 60)

print(f"Validation samples: {len(labels)}")

print("\nMetrics calculated directly from saved model:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nPrediction distribution:")

unique, counts = np.unique(
    predictions,
    return_counts=True
)

for label, count in zip(unique, counts):
    print(
        f"Label {label}: {count}"
    )

print("\nSubject-level predictions:")
print("-" * 60)

for i, subject in enumerate(subjects):

    print(
        f"{subject}: "
        f"Actual={labels[i]} | "
        f"Predicted={predictions[i]} | "
        f"P(Label 0)={probabilities[i][0].item():.4f} | "
        f"P(Label 1)={probabilities[i][1].item():.4f}"
    )