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

from dataset import MultimodalSubjectDataset


# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = "dataset/raw/PHQ/subjects.csv"
TRAIN_FACE_DIR = "dataset/processed/face_features"
TRAIN_AUDIO_DIR = "dataset/processed/audio_features"

OUTPUT_DIR = Path("models/cross_validation")


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_dataset(csv_path, face_dir, audio_dir):

    dataset = MultimodalSubjectDataset(
        csv_path=csv_path,
        face_dir=face_dir,
        audio_dir=audio_dir,
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

    return faces, audios, labels, subjects


# ============================================================
# CREATE MODEL
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
                random_state=42
            )
        )
    ])


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("5-FOLD SUBJECT-LEVEL CROSS-VALIDATION")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading training data...")

    (
        X_face,
        X_audio,
        y,
        subjects
    ) = load_dataset(
        TRAIN_CSV,
        TRAIN_FACE_DIR,
        TRAIN_AUDIO_DIR
    )

    print(
        f"Subjects loaded: {len(y)}"
    )

    print(
        f"Face features : {X_face.shape}"
    )

    print(
        f"Audio features: {X_audio.shape}"
    )

    print(
        f"Class distribution: "
        f"{np.bincount(y)}"
    )

    # ========================================================
    # CHECK THAT EACH SUBJECT IS UNIQUE
    # ========================================================

    if len(subjects) != len(set(subjects)):

        raise ValueError(
            "Duplicate subject IDs detected."
        )

    print(
        "\n✓ All subject IDs are unique."
    )

    # ========================================================
    # CREATE MULTIMODAL FEATURES
    # ========================================================

    X_fusion = np.concatenate(
        [
            X_face,
            X_audio
        ],
        axis=1
    )

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    experiments = {

        "Face Only": X_face,

        "Audio Only": X_audio,

        "Face + Audio": X_fusion
    }

    all_results = {}

    # ========================================================
    # RUN EACH EXPERIMENT
    # ========================================================

    for experiment_name, X in experiments.items():

        print("\n")
        print("=" * 70)
        print(f"EXPERIMENT: {experiment_name}")
        print("=" * 70)

        fold_results = []

        for fold_number, (
            train_indices,
            val_indices
        ) in enumerate(
            cv.split(X, y),
            start=1
        ):

            X_train = X[train_indices]
            X_val = X[val_indices]

            y_train = y[train_indices]
            y_val = y[val_indices]

            train_subjects = [
                subjects[index]
                for index in train_indices
            ]

            val_subjects = [
                subjects[index]
                for index in val_indices
            ]

            # ------------------------------------------------
            # Verify no subject overlap
            # ------------------------------------------------

            overlap = (
                set(train_subjects)
                .intersection(
                    set(val_subjects)
                )
            )

            if overlap:

                raise ValueError(
                    f"Subject leakage detected "
                    f"in fold {fold_number}: "
                    f"{overlap}"
                )

            # ------------------------------------------------
            # Create fresh model
            # ------------------------------------------------

            model = create_model()

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            predictions = model.predict(
                X_val
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

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

            fold_results.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            print(
                f"\nFold {fold_number}"
            )

            print(
                f"Training subjects   : "
                f"{len(train_indices)}"
            )

            print(
                f"Validation subjects : "
                f"{len(val_indices)}"
            )

            print(
                f"Accuracy  : {accuracy:.4f}"
            )

            print(
                f"Precision : {precision:.4f}"
            )

            print(
                f"Recall    : {recall:.4f}"
            )

            print(
                f"F1        : {f1:.4f}"
            )

        # ====================================================
        # CALCULATE MEAN + STANDARD DEVIATION
        # ====================================================

        accuracies = np.array([
            result["accuracy"]
            for result in fold_results
        ])

        precisions = np.array([
            result["precision"]
            for result in fold_results
        ])

        recalls = np.array([
            result["recall"]
            for result in fold_results
        ])

        f1_scores = np.array([
            result["f1"]
            for result in fold_results
        ])

        summary = {

            "accuracy_mean": accuracies.mean(),
            "accuracy_std": accuracies.std(),

            "precision_mean": precisions.mean(),
            "precision_std": precisions.std(),

            "recall_mean": recalls.mean(),
            "recall_std": recalls.std(),

            "f1_mean": f1_scores.mean(),
            "f1_std": f1_scores.std()
        }

        all_results[experiment_name] = summary

        print("\n")
        print("-" * 70)
        print(f"{experiment_name} - CROSS-VALIDATION SUMMARY")
        print("-" * 70)

        print(
            f"Accuracy : "
            f"{summary['accuracy_mean']:.4f} "
            f"+/- "
            f"{summary['accuracy_std']:.4f}"
        )

        print(
            f"Precision: "
            f"{summary['precision_mean']:.4f} "
            f"+/- "
            f"{summary['precision_std']:.4f}"
        )

        print(
            f"Recall   : "
            f"{summary['recall_mean']:.4f} "
            f"+/- "
            f"{summary['recall_std']:.4f}"
        )

        print(
            f"F1       : "
            f"{summary['f1_mean']:.4f} "
            f"+/- "
            f"{summary['f1_std']:.4f}"
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL CROSS-VALIDATION COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<20}"
        f"{'Accuracy':<18}"
        f"{'Precision':<18}"
        f"{'Recall':<18}"
        f"{'F1':<18}"
    )

    print("-" * 90)

    for name, result in all_results.items():

        print(
            f"{name:<20}"
            f"{result['accuracy_mean']:.4f} "
            f"+/- {result['accuracy_std']:.4f}   "
            f"{result['precision_mean']:.4f} "
            f"+/- {result['precision_std']:.4f}   "
            f"{result['recall_mean']:.4f} "
            f"+/- {result['recall_std']:.4f}   "
            f"{result['f1_mean']:.4f} "
            f"+/- {result['f1_std']:.4f}"
        )

    # ========================================================
    # RANK BY F1
    # ========================================================

    ranking = sorted(
        all_results.items(),
        key=lambda item: item[1]["f1_mean"],
        reverse=True
    )

    print("\n")
    print("=" * 70)
    print("RANKING BY MEAN F1")
    print("=" * 70)

    for rank, (
        name,
        result
    ) in enumerate(
        ranking,
        start=1
    ):

        print(
            f"{rank}. "
            f"{name:<18} "
            f"Mean F1 = "
            f"{result['f1_mean']:.4f} "
            f"+/- "
            f"{result['f1_std']:.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = (
        OUTPUT_DIR /
        "cross_validation_results.joblib"
    )

    joblib.dump(
        all_results,
        results_path
    )

    print(
        f"\nResults saved to: "
        f"{results_path}"
    )

    print("\nCross-validation completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()