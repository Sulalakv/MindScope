from pathlib import Path

import numpy as np
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

from dataset import MultimodalSubjectDataset


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent

TRAIN_CSV = DATA_ROOT / "dataset/raw/PHQ/subjects.csv"
TRAIN_FACE_DIR = DATA_ROOT / "dataset/processed/face_features"
TRAIN_AUDIO_DIR = DATA_ROOT / "dataset/processed/audio_features"

VAL_CSV = DATA_ROOT / "dataset_val/raw/PHQ/val.csv"
VAL_FACE_DIR = DATA_ROOT / "dataset_val/features/face_features"
VAL_AUDIO_DIR = DATA_ROOT / "dataset_val/features/audio_mfcc"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(csv_path, face_dir, audio_dir):

    dataset = MultimodalSubjectDataset(
        csv_path=str(csv_path),
        face_dir=str(face_dir),
        audio_dir=str(audio_dir),
    )

    faces = []
    audios = []
    labels = []
    subjects = []

    for sample in dataset.samples:
        subjects.append(sample["subject_id"])
        faces.append(sample["face"])
        audios.append(sample["audio"])
        labels.append(sample["label"])

    faces = np.asarray(faces, dtype=np.float32)
    audios = np.asarray(audios, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    return faces, audios, labels, subjects


# ============================================================
# MODEL
# ============================================================

def create_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="linear",
                class_weight="balanced",
                random_state=42,
            )
        ),
    ])


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VALIDATION-ONLY MODEL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading training data...")

    (
        X_train_face,
        X_train_audio,
        y_train,
        train_subjects,
    ) = load_dataset(
        TRAIN_CSV,
        TRAIN_FACE_DIR,
        TRAIN_AUDIO_DIR,
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("Loading validation data...")

    (
        X_val_face,
        X_val_audio,
        y_val,
        val_subjects,
    ) = load_dataset(
        VAL_CSV,
        VAL_FACE_DIR,
        VAL_AUDIO_DIR,
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA INFORMATION")
    print("=" * 70)

    print(f"\nTraining subjects   : {len(train_subjects)}")
    print(f"Validation subjects : {len(val_subjects)}")

    print("\nFeature shapes:")
    print(f"Training face   : {X_train_face.shape}")
    print(f"Training audio  : {X_train_audio.shape}")
    print(f"Validation face : {X_val_face.shape}")
    print(f"Validation audio: {X_val_audio.shape}")

    print("\nClass distribution:")
    print(f"Training   : {np.bincount(y_train)}")
    print(f"Validation : {np.bincount(y_val)}")

    # --------------------------------------------------------
    # Subject overlap
    # --------------------------------------------------------

    train_set = set(train_subjects)
    val_set = set(val_subjects)

    overlap = train_set.intersection(val_set)

    print("\nSubject overlap:")
    print(f"Train ∩ Validation: {overlap}")

    if overlap:
        raise ValueError(
            "Subject overlap detected. "
            "Validation experiment cannot continue."
        )

    # --------------------------------------------------------
    # Feature-level fusion
    # --------------------------------------------------------

    X_train_fusion = np.concatenate(
        [X_train_face, X_train_audio],
        axis=1,
    )

    X_val_fusion = np.concatenate(
        [X_val_face, X_val_audio],
        axis=1,
    )

    # --------------------------------------------------------
    # Experiment 1: Face only
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. FACE ONLY")
    print("=" * 70)

    face_model = create_model()

    face_model.fit(
        X_train_face,
        y_train,
    )

    face_metrics = evaluate_model(
        face_model,
        X_val_face,
        y_val,
    )

    # --------------------------------------------------------
    # Experiment 2: Audio only
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. AUDIO ONLY")
    print("=" * 70)

    audio_model = create_model()

    audio_model.fit(
        X_train_audio,
        y_train,
    )

    audio_metrics = evaluate_model(
        audio_model,
        X_val_audio,
        y_val,
    )

    # --------------------------------------------------------
    # Experiment 3: Feature-level fusion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. FEATURE-LEVEL FUSION")
    print("=" * 70)

    fusion_model = create_model()

    fusion_model.fit(
        X_train_fusion,
        y_train,
    )

    fusion_metrics = evaluate_model(
        fusion_model,
        X_val_fusion,
        y_val,
    )

    # --------------------------------------------------------
    # Experiment 4: Late fusion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. 50/50 LATE FUSION")
    print("=" * 70)

    # Separate probability-producing SVMs.
    face_probability_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="linear",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )
        ),
    ])

    audio_probability_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="linear",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )
        ),
    ])

    face_probability_model.fit(
        X_train_face,
        y_train,
    )

    audio_probability_model.fit(
        X_train_audio,
        y_train,
    )

    face_probabilities = face_probability_model.predict_proba(
        X_val_face
    )[:, 1]

    audio_probabilities = audio_probability_model.predict_proba(
        X_val_audio
    )[:, 1]

    fused_probabilities = (
        0.5 * face_probabilities
        + 0.5 * audio_probabilities
    )

    late_fusion_predictions = (
        fused_probabilities >= 0.5
    ).astype(int)

    late_fusion_metrics = {
        "accuracy": accuracy_score(
            y_val,
            late_fusion_predictions,
        ),
        "precision": precision_score(
            y_val,
            late_fusion_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_val,
            late_fusion_predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_val,
            late_fusion_predictions,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_val,
            late_fusion_predictions,
        ),
    }

    # --------------------------------------------------------
    # Print detailed results
    # --------------------------------------------------------

    results = {
        "Face Only": face_metrics,
        "Audio Only": audio_metrics,
        "Feature-level Fusion": fusion_metrics,
        "50/50 Late Fusion": late_fusion_metrics,
    }

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for name, metrics in results.items():

        print(f"\n{name}")
        print("-" * 50)

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score : {metrics['f1']:.4f}"
        )

        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RANKING BY F1")
    print("=" * 70)

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["f1"],
        reverse=True,
    )

    for rank, (name, metrics) in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{name:<24} "
            f"F1={metrics['f1']:.4f} "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # Late-fusion subject predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LATE FUSION SUBJECT PREDICTIONS")
    print("=" * 70)

    print(
        f"{'Subject':<10}"
        f"{'Actual':<10}"
        f"{'Face P':<12}"
        f"{'Audio P':<12}"
        f"{'Fused P':<12}"
        f"{'Predicted':<10}"
    )

    print("-" * 70)

    for subject, actual, face_p, audio_p, fused_p, prediction in zip(
        val_subjects,
        y_val,
        face_probabilities,
        audio_probabilities,
        fused_probabilities,
        late_fusion_predictions,
    ):

        print(
            f"{subject:<10}"
            f"{actual:<10}"
            f"{face_p:<12.4f}"
            f"{audio_p:<12.4f}"
            f"{fused_p:<12.4f}"
            f"{prediction:<10}"
        )

    print("\n" + "=" * 70)
    print("VALIDATION-ONLY EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nIMPORTANT:")
    print("- Final test set was NOT loaded.")
    print("- Final test set was NOT evaluated.")
    print("- Models were trained only on 83 training subjects.")
    print("- All comparisons were performed on 14 validation subjects.")


if __name__ == "__main__":
    main()