from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

TRAIN_CSV = ROOT / "dataset/raw/PHQ/subjects.csv"
VAL_CSV = ROOT / "dataset_val/raw/PHQ/val.csv"
TEST_CSV = ROOT / "dataset_test/raw/PHQ/test.csv"

TRAIN_FACE = ROOT / "dataset/processed/face_features"
TRAIN_AUDIO = ROOT / "dataset/processed/audio_features"

VAL_FACE = ROOT / "dataset_val/features/face_agg"
VAL_AUDIO = ROOT / "dataset_val/features/audio_mfcc"

TEST_FACE = ROOT / "dataset_test/features/face_agg"
TEST_AUDIO = ROOT / "dataset_test/features/audio_mfcc"

MODEL_DIR = ROOT / "models/baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 150
PATIENCE = 20


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("BASELINE MULTIMODAL FUSION TRAINING")
print("=" * 60)
print(f"Device: {DEVICE}")


# ============================================================
# LOAD SUBJECT DATA
# ============================================================

def load_split(csv_path, face_dir, audio_dir):
    df = pd.read_csv(csv_path)

    face_features = []
    audio_features = []
    labels = []
    subjects = []

    for _, row in df.iterrows():

        subject_id = str(row["subject_id"])
        label = int(row["Label"])

        face_path = face_dir / f"{subject_id}.npy"
        audio_path = audio_dir / f"{subject_id}.npy"

        if not face_path.exists():
            raise FileNotFoundError(
                f"Missing face feature: {face_path}"
            )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Missing audio feature: {audio_path}"
            )

        face = np.load(face_path)
        audio = np.load(audio_path)

        if face.shape != (1280,):
            raise ValueError(
                f"{subject_id}: expected face shape (1280,), "
                f"got {face.shape}"
            )

        if audio.shape != (40,):
            raise ValueError(
                f"{subject_id}: expected audio shape (40,), "
                f"got {audio.shape}"
            )

        face_features.append(face)
        audio_features.append(audio)
        labels.append(label)
        subjects.append(subject_id)

    return (
        np.asarray(face_features, dtype=np.float32),
        np.asarray(audio_features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        subjects,
    )


# ============================================================
# LOAD TRAIN / VALIDATION / TEST
# ============================================================

print("\nLoading datasets...")

X_face_train, X_audio_train, y_train, train_subjects = load_split(
    TRAIN_CSV,
    TRAIN_FACE,
    TRAIN_AUDIO,
)

X_face_val, X_audio_val, y_val, val_subjects = load_split(
    VAL_CSV,
    VAL_FACE,
    VAL_AUDIO,
)

X_face_test, X_audio_test, y_test, test_subjects = load_split(
    TEST_CSV,
    TEST_FACE,
    TEST_AUDIO,
)

print(f"Training subjects   : {len(y_train)}")
print(f"Validation subjects : {len(y_val)}")
print(f"Testing subjects    : {len(y_test)}")


# ============================================================
# SCALE FEATURES
# IMPORTANT:
# FIT ONLY ON TRAINING DATA
# ============================================================

face_scaler = StandardScaler()
audio_scaler = StandardScaler()

X_face_train = face_scaler.fit_transform(X_face_train)
X_face_val = face_scaler.transform(X_face_val)
X_face_test = face_scaler.transform(X_face_test)

X_audio_train = audio_scaler.fit_transform(X_audio_train)
X_audio_val = audio_scaler.transform(X_audio_val)
X_audio_test = audio_scaler.transform(X_audio_test)


# ============================================================
# SAVE SCALERS
# ============================================================

import joblib

joblib.dump(
    face_scaler,
    MODEL_DIR / "face_scaler.pkl"
)

joblib.dump(
    audio_scaler,
    MODEL_DIR / "audio_scaler.pkl"
)


# ============================================================
# CONVERT TO TENSORS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_face_train, dtype=torch.float32),
    torch.tensor(X_audio_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)

val_dataset = TensorDataset(
    torch.tensor(X_face_val, dtype=torch.float32),
    torch.tensor(X_audio_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)

test_dataset = TensorDataset(
    torch.tensor(X_face_test, dtype=torch.float32),
    torch.tensor(X_audio_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# BASELINE MODEL
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


model = BaselineFusionModel().to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(y_train)

class_weights = len(y_train) / (
    len(class_counts) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)

print("\nClass counts:")
print(f"Label 0: {class_counts[0]}")
print(f"Label 1: {class_counts[1]}")

print("\nClass weights:")
print(class_weights)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
    }


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for face, audio, labels in loader:

            face = face.to(DEVICE)
            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(face, audio)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item() * len(labels)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = total_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    metrics["loss"] = average_loss

    return metrics


# ============================================================
# TRAINING
# ============================================================

history = {

    "train_loss": [],
    "val_loss": [],

    "train_accuracy": [],
    "val_accuracy": [],

    "train_precision": [],
    "val_precision": [],

    "train_recall": [],
    "val_recall": [],

    "train_f1": [],
    "val_f1": [],
}


best_val_f1 = -1
best_epoch = 0
patience_counter = 0

best_model_path = MODEL_DIR / "best_model.pt"


print("\n")
print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)


for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    total_train_loss = 0

    train_labels = []
    train_predictions = []

    for face, audio, labels in train_loader:

        face = face.to(DEVICE)
        audio = audio.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            face,
            audio
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item() * len(labels)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_labels.extend(
            labels.detach().cpu().numpy()
        )

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

    train_loss = (
        total_train_loss /
        len(train_loader.dataset)
    )

    train_metrics = calculate_metrics(
        train_labels,
        train_predictions
    )

    val_metrics = evaluate(
        val_loader
    )

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_metrics["loss"]
    )

    history["train_accuracy"].append(
        train_metrics["accuracy"]
    )

    history["val_accuracy"].append(
        val_metrics["accuracy"]
    )

    history["train_precision"].append(
        train_metrics["precision"]
    )

    history["val_precision"].append(
        val_metrics["precision"]
    )

    history["train_recall"].append(
        train_metrics["recall"]
    )

    history["val_recall"].append(
        val_metrics["recall"]
    )

    history["train_f1"].append(
        train_metrics["f1"]
    )

    history["val_f1"].append(
        val_metrics["f1"]
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Train F1: {train_metrics['f1']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL USING VALIDATION F1
    # --------------------------------------------------------

    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = val_metrics["f1"]

        best_epoch = epoch

        patience_counter = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

    else:

        patience_counter += 1

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    MODEL_DIR / "training_history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)


# ============================================================
# FINAL VALIDATION
# ============================================================

final_val_metrics = evaluate(
    val_loader
)


# ============================================================
# FINAL TEST
# IMPORTANT:
# TEST IS USED ONLY NOW
# ============================================================

final_test_metrics = evaluate(
    test_loader
)


# ============================================================
# TEST PREDICTIONS + CONFUSION MATRIX
# ============================================================

model.eval()

test_predictions = []
test_probabilities = []

with torch.no_grad():

    for face, audio, labels in test_loader:

        face = face.to(DEVICE)
        audio = audio.to(DEVICE)

        outputs = model(
            face,
            audio
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )


cm = confusion_matrix(
    y_test,
    test_predictions
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results = {

    "best_epoch": best_epoch,

    "best_validation_f1": best_val_f1,

    "validation": final_val_metrics,

    "test": final_test_metrics,

    "test_confusion_matrix": cm.tolist(),

    "test_subjects": test_subjects,

    "test_predictions": [
        int(x)
        for x in test_predictions
    ],

    "test_probabilities": [
        float(x)
        for x in test_probabilities
    ],
}


with open(
    MODEL_DIR / "results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE TRAINING COMPLETE")
print("=" * 60)

print(f"\nBest epoch: {best_epoch}")

print("\nValidation:")
print(
    f"Accuracy : {final_val_metrics['accuracy']:.4f}"
)
print(
    f"Precision: {final_val_metrics['precision']:.4f}"
)
print(
    f"Recall   : {final_val_metrics['recall']:.4f}"
)
print(
    f"F1       : {final_val_metrics['f1']:.4f}"
)

print("\nTest:")
print(
    f"Accuracy : {final_test_metrics['accuracy']:.4f}"
)
print(
    f"Precision: {final_test_metrics['precision']:.4f}"
)
print(
    f"Recall   : {final_test_metrics['recall']:.4f}"
)
print(
    f"F1       : {final_test_metrics['f1']:.4f}"
)

print("\nTest Confusion Matrix:")
print(cm)

print("\nSaved files:")
print(f"Model   : {best_model_path}")
print(f"History : {MODEL_DIR / 'training_history.json'}")
print(f"Results : {MODEL_DIR / 'results.json'}")

print("\nBaseline experiment finished.")