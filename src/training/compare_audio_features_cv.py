from pathlib import Path

import numpy as np
import pandas as pd

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
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent

TRAIN_CSV = (
    DATA_ROOT
    / "dataset/raw/PHQ/subjects.csv"
)

TRAIN_AUDIO_40_DIR = (
    DATA_ROOT
    / "dataset/processed/audio_features"
)

TRAIN_AUDIO_80_DIR = (
    DATA_ROOT
    / "dataset/processed/audio_mfcc_stats"
)


# ============================================================
# LOAD AUDIO FEATURES DIRECTLY
# ============================================================

def load_audio_features(audio_dir):
    """
    Load subject-level audio features directly
    from .npy files.

    This function intentionally does not use
    MultimodalSubjectDataset because that loader
    currently expects exactly 40-D audio features.
    """

    metadata = pd.read_csv(TRAIN_CSV)

    features = []
    labels = []
    subjects = []

    for _, row in metadata.iterrows():

        subject_id = str(
            row["subject_id"]
        )

        label = int(
            row["Label"]
        )

        feature_file = (
            Path(audio_dir)
            / f"{subject_id}.npy"
        )

        if not feature_file.exists():
            raise FileNotFoundError(
                f"Missing audio feature: "
                f"{feature_file}"
            )

        feature = np.load(
            feature_file
        ).astype(np.float32)

        if feature.ndim != 1:
            raise ValueError(
                f"{subject_id}: expected "
                f"1-D feature, got "
                f"{feature.shape}"
            )

        features.append(feature)
        labels.append(label)
        subjects.append(subject_id)

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    return (
        features,
        labels,
        subjects,
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            SVC(
                kernel="linear",
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_features(
    X,
    y,
    feature_name,
):

    print("\n" + "=" * 70)
    print(feature_name)
    print("=" * 70)

    print(
        f"Feature shape: {X.shape}"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        model = create_model()

        # Scaler is fitted ONLY on
        # the training part of this fold.
        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_val
        )

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

        accuracies.append(
            accuracy
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

    print("\nOverall:")

    print(
        f"Accuracy : "
        f"{np.mean(accuracies):.4f} "
        f"+/- {np.std(accuracies):.4f}"
    )

    print(
        f"Precision: "
        f"{np.mean(precisions):.4f} "
        f"+/- {np.std(precisions):.4f}"
    )

    print(
        f"Recall   : "
        f"{np.mean(recalls):.4f} "
        f"+/- {np.std(recalls):.4f}"
    )

    print(
        f"F1       : "
        f"{np.mean(f1_scores):.4f} "
        f"+/- {np.std(f1_scores):.4f}"
    )

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1_scores),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO FEATURE REPRESENTATION COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Load 40-D features
    # --------------------------------------------------------

    print(
        "\nLoading 40-D MFCC features..."
    )

    (
        X_40,
        y_40,
        subjects_40,
    ) = load_audio_features(
        TRAIN_AUDIO_40_DIR
    )

    print(
        f"Loaded {len(subjects_40)} subjects."
    )

    # --------------------------------------------------------
    # Load 80-D features
    # --------------------------------------------------------

    print(
        "\nLoading 80-D MFCC statistical features..."
    )

    (
        X_80,
        y_80,
        subjects_80,
    ) = load_audio_features(
        TRAIN_AUDIO_80_DIR
    )

    print(
        f"Loaded {len(subjects_80)} subjects."
    )

    # --------------------------------------------------------
    # Verify dataset consistency
    # --------------------------------------------------------

    print(
        "\nDataset verification:"
    )

    print(
        f"Subjects: "
        f"{len(subjects_40)}"
    )

    print(
        f"40-D shape: "
        f"{X_40.shape}"
    )

    print(
        f"80-D shape: "
        f"{X_80.shape}"
    )

    subjects_match = (
        subjects_40 == subjects_80
    )

    labels_match = np.array_equal(
        y_40,
        y_80,
    )

    print(
        f"Subjects match: "
        f"{subjects_match}"
    )

    print(
        f"Labels match: "
        f"{labels_match}"
    )

    if not subjects_match:
        raise ValueError(
            "Subject ordering differs."
        )

    if not labels_match:
        raise ValueError(
            "Labels differ."
        )

    # --------------------------------------------------------
    # Evaluate 40-D
    # --------------------------------------------------------

    result_40 = evaluate_features(
        X_40,
        y_40,
        "40-D MFCC MEAN",
    )

    # --------------------------------------------------------
    # Evaluate 80-D
    # --------------------------------------------------------

    result_80 = evaluate_features(
        X_80,
        y_80,
        "80-D MFCC MEAN + STANDARD DEVIATION",
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        f"\n{'Metric':<15}"
        f"{'40-D':<15}"
        f"{'80-D':<15}"
    )

    print("-" * 45)

    print(
        f"{'Accuracy':<15}"
        f"{result_40['accuracy']:<15.4f}"
        f"{result_80['accuracy']:<15.4f}"
    )

    print(
        f"{'Precision':<15}"
        f"{result_40['precision']:<15.4f}"
        f"{result_80['precision']:<15.4f}"
    )

    print(
        f"{'Recall':<15}"
        f"{result_40['recall']:<15.4f}"
        f"{result_80['recall']:<15.4f}"
    )

    print(
        f"{'F1':<15}"
        f"{result_40['f1']:<15.4f}"
        f"{result_80['f1']:<15.4f}"
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 70
    )

    difference = (
        result_80["f1"]
        - result_40["f1"]
    )

    print(
        f"\nF1 difference "
        f"(80-D - 40-D): "
        f"{difference:+.4f}"
    )

    if difference > 0:

        print(
            "\n80-D representation performs "
            "better in mean F1."
        )

    elif difference < 0:

        print(
            "\n40-D representation performs "
            "better in mean F1."
        )

    else:

        print(
            "\nBoth representations have "
            "the same mean F1."
        )

    print(
        "\nNo validation data was used."
    )

    print(
        "No test data was used."
    )

    print(
        "Only the 83 training subjects were "
        "used for this experiment."
    )


if __name__ == "__main__":
    main()