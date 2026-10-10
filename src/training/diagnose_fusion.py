from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dataset import MultimodalSubjectDataset


# ============================================================
# PATHS
# ============================================================

VAL_CSV = "dataset_val/raw/PHQ/val.csv"

VAL_FACE_DIR = "dataset_val/features/face_features"
VAL_AUDIO_DIR = "dataset_val/features/audio_mfcc"

OUTPUT_DIR = Path("models/fusion")


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# ============================================================
# MODEL
# Must match train_fusion.py exactly
# ============================================================

DROPOUT = 0.4


class FusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.face_branch = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.audio_branch = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 2),
        )

    def forward(self, face, audio):

        face_features = self.face_branch(face)

        audio_features = self.audio_branch(audio)

        fused = torch.cat(
            [face_features, audio_features],
            dim=1
        )

        return self.classifier(fused)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

val_dataset = MultimodalSubjectDataset(
    csv_path=VAL_CSV,
    face_dir=VAL_FACE_DIR,
    audio_dir=VAL_AUDIO_DIR,
)

print(f"Validation subjects: {len(val_dataset)}")


# ============================================================
# EXTRACT DATA
# ============================================================

faces = []
audios = []
labels = []
subject_ids = []


for sample in val_dataset.samples:

    subject_ids.append(sample["subject_id"])

    faces.append(sample["face"])

    audios.append(sample["audio"])

    labels.append(sample["label"])


X_val_face = np.asarray(faces, dtype=np.float32)
X_val_audio = np.asarray(audios, dtype=np.float32)
y_val = np.asarray(labels, dtype=np.int64)


print(f"Face features shape : {X_val_face.shape}")
print(f"Audio features shape: {X_val_audio.shape}")
print(f"Labels shape        : {y_val.shape}")


# ============================================================
# LOAD TRAINING SCALERS
# Important:
# We use the scalers fitted on TRAINING data.
# We do NOT fit anything on validation data.
# ============================================================

print("\nLoading training scalers...")

face_scaler = joblib.load(
    OUTPUT_DIR / "face_scaler.joblib"
)

audio_scaler = joblib.load(
    OUTPUT_DIR / "audio_scaler.joblib"
)


X_val_face = face_scaler.transform(X_val_face)

X_val_audio = audio_scaler.transform(X_val_audio)


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

face_tensor = torch.tensor(
    X_val_face,
    dtype=torch.float32,
).to(DEVICE)

audio_tensor = torch.tensor(
    X_val_audio,
    dtype=torch.float32,
).to(DEVICE)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model_path = OUTPUT_DIR / "best_fusion_model.pth"

print("\nLoading best fusion model...")

model = FusionModel().to(DEVICE)

checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(
    f"Best epoch from checkpoint: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Best validation F1 saved: "
    f"{checkpoint.get('best_val_f1', 'unknown')}"
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

with torch.no_grad():

    logits = model(
        face_tensor,
        audio_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = torch.argmax(
        logits,
        dim=1
    )


predictions = predictions.cpu().numpy()

probabilities = probabilities.cpu().numpy()


# Probability of class 1 = Depressed
prob_depressed = probabilities[:, 1]


# ============================================================
# PRINT SUBJECT-LEVEL PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION PREDICTIONS")
print("=" * 70)

print(
    f"{'Subject':<10}"
    f"{'Actual':<10}"
    f"{'Predicted':<12}"
    f"{'P(Depressed)':<15}"
)

print("-" * 70)


for subject, actual, predicted, probability in zip(
    subject_ids,
    y_val,
    predictions,
    prob_depressed,
):

    print(
        f"{subject:<10}"
        f"{actual:<10}"
        f"{predicted:<12}"
        f"{probability:<15.4f}"
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    predictions
)

precision = precision_score(
    y_val,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_val,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_val,
    predictions,
    zero_division=0
)


print("\n")
print("=" * 70)
print("VALIDATION METRICS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    predictions
)


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

unique, counts = np.unique(
    predictions,
    return_counts=True
)


print("\n")
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)


for label, count in zip(unique, counts):

    label_name = (
        "Non-Depressed"
        if label == 0
        else "Depressed"
    )

    print(
        f"{label_name}: {count}"
    )


# ============================================================
# PROBABILITY RANGE
# ============================================================

print("\n")
print("=" * 70)
print("PROBABILITY INFORMATION")
print("=" * 70)

print(
    f"Minimum P(Depressed): "
    f"{prob_depressed.min():.4f}"
)

print(
    f"Maximum P(Depressed): "
    f"{prob_depressed.max():.4f}"
)

print(
    f"Mean P(Depressed): "
    f"{prob_depressed.mean():.4f}"
)


print("\nDiagnostic completed.")