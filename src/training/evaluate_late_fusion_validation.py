# src/training/evaluate_late_fusion_validation.py

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_CSV = PROJECT_ROOT / "dataset" / "raw" / "PHQ" / "subjects.csv"

VAL_CSV = PROJECT_ROOT / "dataset_val" / "raw" / "PHQ" / "val.csv"

TRAIN_FACE_DIR = (
    PROJECT_ROOT / "dataset" / "processed" / "face_features"
)

VAL_FACE_DIR = (
    PROJECT_ROOT / "dataset_val" / "features" / "face_features"
)

TRAIN_AUDIO_DIR = (
    PROJECT_ROOT / "dataset" / "processed" / "audio_features"
)

VAL_AUDIO_DIR = (
    PROJECT_ROOT / "dataset_val" / "features" / "audio_mfcc"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
FACE_WEIGHT = 0.5
AUDIO_WEIGHT = 0.5
THRESHOLD = 0.5


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(subject_ids, face_dir, audio_dir):
    """
    Load one face feature vector and one audio feature vector
    for every subject.
    """

    face_features = []
    audio_features = []

    valid_subjects = []

    for subject_id in subject_ids:

        face_path = face_dir / f"{subject_id}.npy"
        audio_path = audio_dir / f"{subject_id}.npy"

        if not face_path.exists():
            print(f"WARNING: Missing face feature: {subject_id}")
            continue

        if not audio_path.exists():
            print(f"WARNING: Missing audio feature: {subject_id}")
            continue

        face = np.load(face_path)
        audio = np.load(audio_path)

        # Flatten in case features have an extra dimension
        face = face.reshape(-1)
        audio = audio.reshape(-1)

        face_features.append(face)
        audio_features.append(audio)
        valid_subjects.append(subject_id)

    return (
        np.asarray(face_features, dtype=np.float32),
        np.asarray(audio_features, dtype=np.float32),
        valid_subjects,
    )


# ============================================================
# CREATE SVM
# ============================================================

def create_svm():
    """
    Linear SVM with scaling and balanced class weights.
    """

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="linear",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LATE FUSION VALIDATION EVALUATION")
    print("=" * 70)

    print("\nLoading CSV files...")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    print(f"Training subjects   : {len(train_df)}")
    print(f"Validation subjects : {len(val_df)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = {"subject_id", "Label"}

    if not required_columns.issubset(train_df.columns):
        raise ValueError(
            f"Training CSV must contain: {required_columns}"
        )

    if not required_columns.issubset(val_df.columns):
        raise ValueError(
            f"Validation CSV must contain: {required_columns}"
        )

    # --------------------------------------------------------
    # Extract subject IDs and labels
    # --------------------------------------------------------

    train_subjects = train_df["subject_id"].astype(str).tolist()
    val_subjects = val_df["subject_id"].astype(str).tolist()

    train_labels_dict = dict(
        zip(
            train_df["subject_id"].astype(str),
            train_df["Label"].astype(int),
        )
    )

    val_labels_dict = dict(
        zip(
            val_df["subject_id"].astype(str),
            val_df["Label"].astype(int),
        )
    )

    # --------------------------------------------------------
    # Load training features
    # --------------------------------------------------------

    print("\nLoading training features...")

    X_train_face, X_train_audio, train_valid_subjects = load_features(
        train_subjects,
        TRAIN_FACE_DIR,
        TRAIN_AUDIO_DIR,
    )

    y_train = np.array(
        [train_labels_dict[s] for s in train_valid_subjects],
        dtype=int,
    )

    # --------------------------------------------------------
    # Load validation features
    # --------------------------------------------------------

    print("Loading validation features...")

    X_val_face, X_val_audio, val_valid_subjects = load_features(
        val_subjects,
        VAL_FACE_DIR,
        VAL_AUDIO_DIR,
    )

    y_val = np.array(
        [val_labels_dict[s] for s in val_valid_subjects],
        dtype=int,
    )

    # --------------------------------------------------------
    # Dataset verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE INFORMATION")
    print("=" * 70)

    print(f"Training face shape   : {X_train_face.shape}")
    print(f"Training audio shape  : {X_train_audio.shape}")
    print(f"Validation face shape : {X_val_face.shape}")
    print(f"Validation audio shape: {X_val_audio.shape}")

    print("\nTraining labels:")
    print(f"  Label 0: {(y_train == 0).sum()}")
    print(f"  Label 1: {(y_train == 1).sum()}")

    print("\nValidation labels:")
    print(f"  Label 0: {(y_val == 0).sum()}")
    print(f"  Label 1: {(y_val == 1).sum()}")

    # --------------------------------------------------------
    # Train FACE model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FACE SVM")
    print("=" * 70)

    face_model = create_svm()

    face_model.fit(
        X_train_face,
        y_train,
    )

    print("✓ Face SVM trained.")

    # --------------------------------------------------------
    # Train AUDIO model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING AUDIO SVM")
    print("=" * 70)

    audio_model = create_svm()

    audio_model.fit(
        X_train_audio,
        y_train,
    )

    print("✓ Audio SVM trained.")

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print("\nGenerating validation probabilities...")

    face_probabilities = face_model.predict_proba(
        X_val_face
    )[:, 1]

    audio_probabilities = audio_model.predict_proba(
        X_val_audio
    )[:, 1]

    # --------------------------------------------------------
    # 50/50 LATE FUSION
    # --------------------------------------------------------

    fused_probabilities = (
        FACE_WEIGHT * face_probabilities
        + AUDIO_WEIGHT * audio_probabilities
    )

    predictions = (
        fused_probabilities >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Per-subject predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION PREDICTIONS")
    print("=" * 70)

    print(
        f"{'Subject':<12}"
        f"{'Actual':<10}"
        f"{'Face P':<12}"
        f"{'Audio P':<12}"
        f"{'Fused P':<12}"
        f"{'Predicted':<10}"
    )

    print("-" * 70)

    for i, subject_id in enumerate(val_valid_subjects):

        print(
            f"{subject_id:<12}"
            f"{y_val[i]:<10}"
            f"{face_probabilities[i]:<12.4f}"
            f"{audio_probabilities[i]:<12.4f}"
            f"{fused_probabilities[i]:<12.4f}"
            f"{predictions[i]:<10}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_val,
        predictions,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("50/50 LATE FUSION VALIDATION RESULTS")
    print("=" * 70)

    print(f"Face weight       : {FACE_WEIGHT}")
    print(f"Audio weight      : {AUDIO_WEIGHT}")
    print(f"Decision threshold: {THRESHOLD}")

    print("\nMetrics:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\n" + "=" * 70)
    print("VALIDATION EVALUATION COMPLETE")
    print("=" * 70)

    print("\nIMPORTANT:")
    print("- The final test set was NOT loaded.")
    print("- The final test set was NOT used.")
    print("- The models were trained only on the 83 training subjects.")
    print("- Validation contains 14 subjects.")
    print("- Late fusion uses 50% face + 50% audio probability.")


if __name__ == "__main__":
    main()