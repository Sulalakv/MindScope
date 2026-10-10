from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = "dataset/raw/PHQ/subjects.csv"

TRAIN_FACE_DIR = "dataset/processed/face_features"
TRAIN_AUDIO_DIR = "dataset/processed/audio_features"

RANDOM_STATE = 42
N_SPLITS = 5

OUTPUT_DIR = Path("models/late_fusion")


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(csv_path, face_dir, audio_dir):
    import pandas as pd

    df = pd.read_csv(csv_path)

    subjects = []
    face_features = []
    audio_features = []
    labels = []

    for _, row in df.iterrows():

        subject_id = str(row["subject_id"]).strip()
        label = int(row["Label"])

        face_path = Path(face_dir) / f"{subject_id}.npy"
        audio_path = Path(audio_dir) / f"{subject_id}.npy"

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

        subjects.append(subject_id)
        face_features.append(face)
        audio_features.append(audio)
        labels.append(label)

    return (
        np.array(subjects),
        np.array(face_features),
        np.array(audio_features),
        np.array(labels),
    )


# ============================================================
# MODEL
# ============================================================

def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="linear",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
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
# CROSS-VALIDATION
# ============================================================

def run_late_fusion():

    print("=" * 70)
    print("5-FOLD SUBJECT-LEVEL LATE FUSION")
    print("=" * 70)

    print("\nLoading training data...")

    subjects, face, audio, labels = load_dataset(
        TRAIN_CSV,
        TRAIN_FACE_DIR,
        TRAIN_AUDIO_DIR,
    )

    print(f"Subjects loaded : {len(subjects)}")
    print(f"Face features   : {face.shape}")
    print(f"Audio features  : {audio.shape}")
    print(f"Class distribution: {np.bincount(labels)}")

    if len(np.unique(subjects)) != len(subjects):
        raise ValueError("Duplicate subject IDs detected.")

    print("\n✓ All subject IDs are unique.")

    # --------------------------------------------------------
    # 5-FOLD SUBJECT-LEVEL CV
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(face, labels),
        start=1,
    ):

        print("\n" + "=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        train_subjects = set(subjects[train_idx])
        val_subjects = set(subjects[val_idx])

        overlap = train_subjects.intersection(val_subjects)

        if overlap:
            raise ValueError(
                f"Subject leakage detected: {overlap}"
            )

        print(f"Training subjects   : {len(train_idx)}")
        print(f"Validation subjects : {len(val_idx)}")
        print("Subject overlap     : None")

        # ----------------------------------------------------
        # Train face model
        # ----------------------------------------------------

        face_model = create_model()

        face_model.fit(
            face[train_idx],
            labels[train_idx],
        )

        face_probability = face_model.predict_proba(
            face[val_idx]
        )[:, 1]

        # ----------------------------------------------------
        # Train audio model
        # ----------------------------------------------------

        audio_model = create_model()

        audio_model.fit(
            audio[train_idx],
            labels[train_idx],
        )

        audio_probability = audio_model.predict_proba(
            audio[val_idx]
        )[:, 1]

        # ----------------------------------------------------
        # Individual predictions
        # ----------------------------------------------------

        face_pred = (
            face_probability >= 0.5
        ).astype(int)

        audio_pred = (
            audio_probability >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Late fusion
        #
        # Average the two probabilities.
        # ----------------------------------------------------

        fused_probability = (
            face_probability + audio_probability
        ) / 2.0

        fused_pred = (
            fused_probability >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        face_metrics = calculate_metrics(
            labels[val_idx],
            face_pred,
        )

        audio_metrics = calculate_metrics(
            labels[val_idx],
            audio_pred,
        )

        fused_metrics = calculate_metrics(
            labels[val_idx],
            fused_pred,
        )

        print("\nFace Only")
        print(
            f"Accuracy  : {face_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision : {face_metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {face_metrics['recall']:.4f}"
        )
        print(
            f"F1        : {face_metrics['f1']:.4f}"
        )

        print("\nAudio Only")
        print(
            f"Accuracy  : {audio_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision : {audio_metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {audio_metrics['recall']:.4f}"
        )
        print(
            f"F1        : {audio_metrics['f1']:.4f}"
        )

        print("\nFace + Audio (Late Fusion)")
        print(
            f"Accuracy  : {fused_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision : {fused_metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {fused_metrics['recall']:.4f}"
        )
        print(
            f"F1        : {fused_metrics['f1']:.4f}"
        )

        results.append({
            "fold": fold,

            "face_accuracy": face_metrics["accuracy"],
            "face_precision": face_metrics["precision"],
            "face_recall": face_metrics["recall"],
            "face_f1": face_metrics["f1"],

            "audio_accuracy": audio_metrics["accuracy"],
            "audio_precision": audio_metrics["precision"],
            "audio_recall": audio_metrics["recall"],
            "audio_f1": audio_metrics["f1"],

            "fusion_accuracy": fused_metrics["accuracy"],
            "fusion_precision": fused_metrics["precision"],
            "fusion_recall": fused_metrics["recall"],
            "fusion_f1": fused_metrics["f1"],
        })

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL LATE-FUSION CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    import pandas as pd

    results_df = pd.DataFrame(results)

    experiments = {
        "Face Only": [
            "face_accuracy",
            "face_precision",
            "face_recall",
            "face_f1",
        ],
        "Audio Only": [
            "audio_accuracy",
            "audio_precision",
            "audio_recall",
            "audio_f1",
        ],
        "Face + Audio": [
            "fusion_accuracy",
            "fusion_precision",
            "fusion_recall",
            "fusion_f1",
        ],
    }

    summary = {}

    for name, columns in experiments.items():

        means = results_df[columns].mean()
        stds = results_df[columns].std(ddof=0)

        summary[name] = {
            "accuracy_mean": means[columns[0]],
            "accuracy_std": stds[columns[0]],

            "precision_mean": means[columns[1]],
            "precision_std": stds[columns[1]],

            "recall_mean": means[columns[2]],
            "recall_std": stds[columns[2]],

            "f1_mean": means[columns[3]],
            "f1_std": stds[columns[3]],
        }

        print(f"\n{name}")
        print("-" * 50)

        print(
            f"Accuracy : "
            f"{means[columns[0]]:.4f} "
            f"+/- {stds[columns[0]]:.4f}"
        )

        print(
            f"Precision: "
            f"{means[columns[1]]:.4f} "
            f"+/- {stds[columns[1]]:.4f}"
        )

        print(
            f"Recall   : "
            f"{means[columns[2]]:.4f} "
            f"+/- {stds[columns[2]]:.4f}"
        )

        print(
            f"F1       : "
            f"{means[columns[3]]:.4f} "
            f"+/- {stds[columns[3]]:.4f}"
        )

    # ========================================================
    # RANKING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RANKING BY MEAN F1")
    print("=" * 70)

    ranking = sorted(
        summary.items(),
        key=lambda x: x[1]["f1_mean"],
        reverse=True,
    )

    for rank, (name, metrics) in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{rank}. {name:<20} "
            f"F1 = {metrics['f1_mean']:.4f} "
            f"+/- {metrics['f1_std']:.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR /
        "late_fusion_results.joblib"
    )

    joblib.dump(
        {
            "fold_results": results_df,
            "summary": summary,
            "ranking": ranking,
        },
        output_path,
    )

    print("\nResults saved to:")
    print(output_path)

    print("\nLate-fusion experiment completed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_late_fusion()