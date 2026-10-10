
from pathlib import Path
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import MultimodalSubjectDataset


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# 2. PATHS
# ============================================================

TRAIN_CSV = "dataset/raw/PHQ/subjects.csv"
TRAIN_FACE_DIR = "dataset/processed/face_features"
TRAIN_AUDIO_DIR = "dataset/processed/audio_features"

VAL_CSV = "dataset_val/raw/PHQ/val.csv"
VAL_FACE_DIR = "dataset_val/features/face_features"
VAL_AUDIO_DIR = "dataset_val/features/audio_mfcc"

TEST_CSV = "dataset_test/raw/PHQ/test.csv"
TEST_FACE_DIR = "dataset_test/features/face_features"
TEST_AUDIO_DIR = "dataset_test/features/audio_mfcc"

OUTPUT_DIR = Path("models/fusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16
MAX_EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DROPOUT = 0.4

PATIENCE = 20
MIN_DELTA = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# 4. LOAD DATA
# ============================================================

train_dataset = MultimodalSubjectDataset(
    csv_path=TRAIN_CSV,
    face_dir=TRAIN_FACE_DIR,
    audio_dir=TRAIN_AUDIO_DIR,
)

val_dataset = MultimodalSubjectDataset(
    csv_path=VAL_CSV,
    face_dir=VAL_FACE_DIR,
    audio_dir=VAL_AUDIO_DIR,
)

test_dataset = MultimodalSubjectDataset(
    csv_path=TEST_CSV,
    face_dir=TEST_FACE_DIR,
    audio_dir=TEST_AUDIO_DIR,
)


# ============================================================
# 5. CONVERT DATASETS TO NUMPY ARRAYS
# ============================================================

def dataset_to_arrays(dataset):
    faces = []
    audios = []
    labels = []
    subjects = []

    for sample in dataset.samples:
        subjects.append(sample["subject_id"])
        faces.append(sample["face"])
        audios.append(sample["audio"])
        labels.append(sample["label"])

    return (
        np.asarray(faces, dtype=np.float32),
        np.asarray(audios, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        subjects,
    )


X_train_face, X_train_audio, y_train, train_subjects = (
    dataset_to_arrays(train_dataset)
)

X_val_face, X_val_audio, y_val, val_subjects = (
    dataset_to_arrays(val_dataset)
)

X_test_face, X_test_audio, y_test, test_subjects = (
    dataset_to_arrays(test_dataset)
)


print("\nDataset sizes:")
print("Train:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))

print("\nClass distribution:")
print("Train:", np.bincount(y_train))
print("Validation:", np.bincount(y_val))
print("Test:", np.bincount(y_test))


# ============================================================
# 6. VERIFY SUBJECT INDEPENDENCE
# ============================================================

train_set = set(train_subjects)
val_set = set(val_subjects)
test_set = set(test_subjects)

train_val_overlap = train_set & val_set
train_test_overlap = train_set & test_set
val_test_overlap = val_set & test_set

print("\nSubject overlap check:")
print("Train ∩ Validation:", train_val_overlap)
print("Train ∩ Test:", train_test_overlap)
print("Validation ∩ Test:", val_test_overlap)

if train_val_overlap or train_test_overlap or val_test_overlap:
    raise RuntimeError(
        "Subject overlap detected between dataset splits."
    )

print("✓ No subject overlap detected.")


# ============================================================
# 7. STANDARDIZATION
# ============================================================
#
# IMPORTANT:
# Scalers are fitted ONLY on the training subjects.
#
# Validation and test data are transformed using the
# training scalers.
# ============================================================

face_scaler = StandardScaler()
audio_scaler = StandardScaler()

X_train_face = face_scaler.fit_transform(X_train_face)
X_val_face = face_scaler.transform(X_val_face)
X_test_face = face_scaler.transform(X_test_face)

X_train_audio = audio_scaler.fit_transform(X_train_audio)
X_val_audio = audio_scaler.transform(X_val_audio)
X_test_audio = audio_scaler.transform(X_test_audio)


joblib.dump(
    face_scaler,
    OUTPUT_DIR / "face_scaler.joblib",
)

joblib.dump(
    audio_scaler,
    OUTPUT_DIR / "audio_scaler.joblib",
)


# ============================================================
# 8. TORCH DATASET
# ============================================================

class FusionDataset(torch.utils.data.Dataset):

    def __init__(self, face, audio, labels):
        self.face = torch.tensor(
            face,
            dtype=torch.float32,
        )

        self.audio = torch.tensor(
            audio,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.face[index],
            self.audio[index],
            self.labels[index],
        )


train_loader = DataLoader(
    FusionDataset(
        X_train_face,
        X_train_audio,
        y_train,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    FusionDataset(
        X_val_face,
        X_val_audio,
        y_val,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    FusionDataset(
        X_test_face,
        X_test_audio,
        y_test,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# 9. MODEL
# ============================================================

class FusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Face: 1280 -> 128
        self.face_branch = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # Audio: 40 -> 128
        self.audio_branch = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # Fusion: 128 + 128 -> 256 -> 64 -> 2
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
            dim=1,
        )

        return self.classifier(fused)


model = FusionModel().to(DEVICE)

print("\nModel:")
print(model)


# ============================================================
# 10. CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(y_train)

if len(class_counts) != 2:
    raise ValueError(
        "Training dataset must contain both classes."
    )

total = len(y_train)

class_weights = total / (
    2 * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
).to(DEVICE)

print("\nClass weights:", class_weights)


# ============================================================
# 11. LOSS + OPTIMIZER
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
# 12. METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


# ============================================================
# 13. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for face, audio, labels in train_loader:

        face = face.to(DEVICE)
        audio = audio.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            face,
            audio,
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    average_loss = (
        total_loss / len(y_train)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    return average_loss, metrics


# ============================================================
# 14. VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for face, audio, labels in val_loader:

            face = face.to(DEVICE)
            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                face,
                audio,
            )

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = (
        total_loss / len(y_val)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    return average_loss, metrics


# ============================================================
# 15. TRAINING LOOP
# ============================================================

history = []

best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

best_model_path = (
    OUTPUT_DIR / "best_fusion_model.pth"
)

print("\nStarting training...\n")

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss, train_metrics = train_one_epoch()

    val_loss, val_metrics = validate()

    history.append(
        {
            "epoch": epoch,

            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],

            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train F1: {train_metrics['f1']:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Best model based ONLY on validation F1.
    # Test data is not touched.
    # --------------------------------------------------------

    if val_metrics["f1"] > (
        best_val_f1 + MIN_DELTA
    ):

        best_val_f1 = val_metrics["f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_val_f1,
                "epoch": best_epoch,
            },
            best_model_path,
        )

        print(
            f"  ✓ Best model saved "
            f"(validation F1={best_val_f1:.4f})"
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}."
        )

        break


# ============================================================
# 16. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = (
    OUTPUT_DIR / "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False,
)

print(
    f"\nTraining history saved to: "
    f"{history_path}"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation F1: "
    f"{best_val_f1:.4f}"
)


# ============================================================
# 17. TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Train Loss",
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()

loss_plot_path = (
    OUTPUT_DIR / "loss_curve.png"
)

plt.savefig(
    loss_plot_path,
    dpi=150,
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_f1"],
    label="Train F1",
)

plt.plot(
    history_df["epoch"],
    history_df["val_f1"],
    label="Validation F1",
)

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Training vs Validation F1")
plt.legend()
plt.tight_layout()

f1_plot_path = (
    OUTPUT_DIR / "f1_curve.png"
)

plt.savefig(
    f1_plot_path,
    dpi=150,
)

plt.close()


# ============================================================
# 18. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# 19. FINAL TEST EVALUATION
# ============================================================
#
# IMPORTANT:
# This is the FIRST time we evaluate the test set.
# It was never used for model selection.
# ============================================================

all_test_predictions = []
all_test_labels = []

with torch.no_grad():

    for face, audio, labels in test_loader:

        face = face.to(DEVICE)
        audio = audio.to(DEVICE)

        outputs = model(
            face,
            audio,
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_test_predictions.extend(
            predictions.cpu().numpy()
        )

        all_test_labels.extend(
            labels.numpy()
        )


test_metrics = calculate_metrics(
    all_test_labels,
    all_test_predictions,
)


print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy : {test_metrics['accuracy']:.4f}"
)

print(
    f"Precision: {test_metrics['precision']:.4f}"
)

print(
    f"Recall   : {test_metrics['recall']:.4f}"
)

print(
    f"F1 Score : {test_metrics['f1']:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        all_test_labels,
        all_test_predictions,
        target_names=[
            "Non-Depressed",
            "Depressed",
        ],
        zero_division=0,
    )
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_test_labels,
    all_test_predictions,
)

print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Test Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["Non-Depressed", "Depressed"],
)

plt.yticks(
    [0, 1],
    ["Non-Depressed", "Depressed"],
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.tight_layout()

cm_path = (
    OUTPUT_DIR / "test_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150,
)

plt.close()


# ============================================================
# 21. SAVE TEST RESULTS
# ============================================================

test_results = {
    "best_epoch": best_epoch,
    "best_validation_f1": best_val_f1,

    "test_accuracy": test_metrics["accuracy"],
    "test_precision": test_metrics["precision"],
    "test_recall": test_metrics["recall"],
    "test_f1": test_metrics["f1"],
}

pd.DataFrame(
    [test_results]
).to_json(
    OUTPUT_DIR / "test_results.json",
    orient="records",
    indent=4,
)


print("\nAll results saved to:")
print(OUTPUT_DIR)

print("\nTraining completed successfully.")

