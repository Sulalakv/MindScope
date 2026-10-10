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

TRAIN_CSV = Path("../dataset/raw/PHQ/subjects.csv")

FACE_DIR = Path("../dataset/processed/face_features")
AUDIO_40_DIR = Path("../dataset/processed/audio_features")
AUDIO_80_DIR = Path("../dataset/processed/audio_mfcc_stats")


# ============================================================
# LOAD DATA
# ============================================================

def load_features():
    df = pd.read_csv(TRAIN_CSV)

    face_features = []
    audio_40_features = []
    audio_80_features = []
    labels = []
    subject_ids = []

    for _, row in df.iterrows():

        subject_id = str(row["subject_id"])
        label = int(row["Label"])

        face_path = FACE_DIR / f"{subject_id}.npy"
        audio_40_path = AUDIO_40_DIR / f"{subject_id}.npy"
        audio_80_path = AUDIO_80_DIR / f"{subject_id}.npy"

        if not face_path.exists():
            raise FileNotFoundError(face_path)

        if not audio_40_path.exists():
            raise FileNotFoundError(audio_40_path)

        if not audio_80_path.exists():
            raise FileNotFoundError(audio_80_path)

        face = np.load(face_path)
        audio_40 = np.load(audio_40_path)
        audio_80 = np.load(audio_80_path)

        face_features.append(face)
        audio_40_features.append(audio_40)
        audio_80_features.append(audio_80)

        labels.append(label)
        subject_ids.append(subject_id)

    return (
        np.asarray(face_features, dtype=np.float32),
        np.asarray(audio_40_features, dtype=np.float32),
        np.asarray(audio_80_features, dtype=np.float32),
        np.asarray(labels),
        subject_ids,
    )


# ============================================================
# MODEL
# ============================================================

def create_svm():

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                kernel="linear",
                class_weight="balanced",
                probability=True,
                random_state=42,
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
# CROSS VALIDATION
# ============================================================

def evaluate_cv(X, y, model_name, n_splits=5):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        model = create_svm()

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        metrics = calculate_metrics(
            y_val,
            predictions,
        )

        fold_results.append(metrics)

        print(
            f"Fold {fold}: "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)

    print("\nOverall:")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        mean = results_df[metric].mean()
        std = results_df[metric].std()

        print(
            f"{metric.capitalize():10s}: "
            f"{mean:.4f} +/- {std:.4f}"
        )

    return {
        metric: (
            results_df[metric].mean(),
            results_df[metric].std(),
        )
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    }


# ============================================================
# LATE FUSION
# ============================================================

def evaluate_late_fusion(
    X_face,
    X_audio,
    y,
    n_splits=5,
):

    print("\n" + "=" * 70)
    print("LATE FUSION — FACE + 80-D AUDIO")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_face, y),
        start=1,
    ):

        face_model = create_svm()
        audio_model = create_svm()

        face_model.fit(
            X_face[train_idx],
            y[train_idx],
        )

        audio_model.fit(
            X_audio[train_idx],
            y[train_idx],
        )

        face_probability = face_model.predict_proba(
            X_face[val_idx]
        )[:, 1]

        audio_probability = audio_model.predict_proba(
            X_audio[val_idx]
        )[:, 1]

        # Equal-weight late fusion
        fused_probability = (
            0.5 * face_probability
            + 0.5 * audio_probability
        )

        predictions = (
            fused_probability >= 0.5
        ).astype(int)

        metrics = calculate_metrics(
            y[val_idx],
            predictions,
        )

        fold_results.append(metrics)

        print(
            f"Fold {fold}: "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)

    print("\nOverall:")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        mean = results_df[metric].mean()
        std = results_df[metric].std()

        print(
            f"{metric.capitalize():10s}: "
            f"{mean:.4f} +/- {std:.4f}"
        )

    return {
        metric: (
            results_df[metric].mean(),
            results_df[metric].std(),
        )
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IMPROVED MULTIMODAL REPRESENTATION COMPARISON")
    print("=" * 70)

    (
        X_face,
        X_audio_40,
        X_audio_80,
        y,
        subject_ids,
    ) = load_features()

    print(f"\nSubjects: {len(y)}")
    print(f"Face features: {X_face.shape}")
    print(f"Audio 40-D:    {X_audio_40.shape}")
    print(f"Audio 80-D:    {X_audio_80.shape}")
    print(f"Labels:        {y.shape}")

    print("\nClass distribution:")
    print(
        f"Label 0: {(y == 0).sum()}"
    )
    print(
        f"Label 1: {(y == 1).sum()}"
    )

    # --------------------------------------------------------
    # 1. FACE ONLY
    # --------------------------------------------------------

    face_results = evaluate_cv(
        X_face,
        y,
        "FACE ONLY — 1280-D",
    )

    # --------------------------------------------------------
    # 2. AUDIO ONLY — 80-D
    # --------------------------------------------------------

    audio_results = evaluate_cv(
        X_audio_80,
        y,
        "AUDIO ONLY — 80-D MFCC MEAN + STD",
    )

    # --------------------------------------------------------
    # 3. FEATURE-LEVEL FUSION
    # --------------------------------------------------------

    X_feature_fusion = np.concatenate(
        [
            X_face,
            X_audio_80,
        ],
        axis=1,
    )

    print(
        f"\nFeature-level fusion shape: "
        f"{X_feature_fusion.shape}"
    )

    feature_fusion_results = evaluate_cv(
        X_feature_fusion,
        y,
        "FEATURE-LEVEL FUSION — 1280 + 80 = 1360-D",
    )

    # --------------------------------------------------------
    # 4. LATE FUSION
    # --------------------------------------------------------

    late_fusion_results = evaluate_late_fusion(
        X_face,
        X_audio_80,
        y,
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Model':35s}"
        f"{'Accuracy':>15s}"
        f"{'Precision':>15s}"
        f"{'Recall':>15s}"
        f"{'F1':>15s}"
    )

    print("-" * 95)

    results = {
        "Face Only": face_results,
        "Audio Only (80-D)": audio_results,
        "Feature Fusion (1360-D)": feature_fusion_results,
        "Late Fusion (50/50)": late_fusion_results,
    }

    for name, result in results.items():

        print(
            f"{name:35s}"
            f"{result['accuracy'][0]:.4f} +/- {result['accuracy'][1]:.4f}"
            f"{result['precision'][0]:.4f} +/- {result['precision'][1]:.4f}"
            f"{result['recall'][0]:.4f} +/- {result['recall'][1]:.4f}"
            f"{result['f1'][0]:.4f} +/- {result['f1'][1]:.4f}"
        )

    print("\n")
    print("IMPORTANT:")
    print("- Only the 83 training subjects were used.")
    print("- Validation and test sets were NOT used.")
    print("- Scaling was fitted separately inside each CV fold.")
    print("- Model selection should be based on CV results.")
    print("- The final test set remains locked.")


if __name__ == "__main__":
    main()