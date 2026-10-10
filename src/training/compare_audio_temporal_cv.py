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

AUDIO_40_DIR = Path("../dataset/processed/audio_features")
AUDIO_80_DIR = Path("../dataset/processed/audio_mfcc_stats")
AUDIO_240_DIR = Path("../dataset/processed/audio_mfcc_temporal")


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    df = pd.read_csv(TRAIN_CSV)

    audio_40 = []
    audio_80 = []
    audio_240 = []
    labels = []

    for _, row in df.iterrows():

        subject_id = str(row["subject_id"])
        label = int(row["Label"])

        path_40 = AUDIO_40_DIR / f"{subject_id}.npy"
        path_80 = AUDIO_80_DIR / f"{subject_id}.npy"
        path_240 = AUDIO_240_DIR / f"{subject_id}.npy"

        if not path_40.exists():
            raise FileNotFoundError(path_40)

        if not path_80.exists():
            raise FileNotFoundError(path_80)

        if not path_240.exists():
            raise FileNotFoundError(path_240)

        feature_40 = np.load(path_40)
        feature_80 = np.load(path_80)
        feature_240 = np.load(path_240)

        if feature_40.shape != (40,):
            raise ValueError(
                f"{subject_id}: expected (40,), "
                f"got {feature_40.shape}"
            )

        if feature_80.shape != (80,):
            raise ValueError(
                f"{subject_id}: expected (80,), "
                f"got {feature_80.shape}"
            )

        if feature_240.shape != (240,):
            raise ValueError(
                f"{subject_id}: expected (240,), "
                f"got {feature_240.shape}"
            )

        audio_40.append(feature_40)
        audio_80.append(feature_80)
        audio_240.append(feature_240)
        labels.append(label)

    return (
        np.asarray(audio_40, dtype=np.float32),
        np.asarray(audio_80, dtype=np.float32),
        np.asarray(audio_240, dtype=np.float32),
        np.asarray(labels),
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    return Pipeline([
        ("scaler", StandardScaler()),
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
# METRICS
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
# CROSS VALIDATION
# ============================================================

def evaluate_representation(
    X,
    y,
    name,
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Feature shape: {X.shape}"
    )

    cv = StratifiedKFold(
        n_splits=5,
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

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_val
        )

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

    results = pd.DataFrame(
        fold_results
    )

    print("\nOverall:")

    summary = {}

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:

        mean = results[metric].mean()
        std = results[metric].std()

        summary[metric] = (
            mean,
            std,
        )

        print(
            f"{metric.capitalize():10s}: "
            f"{mean:.4f} +/- {std:.4f}"
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO REPRESENTATION TEMPORAL COMPARISON")
    print("=" * 70)

    (
        audio_40,
        audio_80,
        audio_240,
        y,
    ) = load_features()

    print(
        f"\nNumber of subjects: {len(y)}"
    )

    print(
        f"40-D features : {audio_40.shape}"
    )

    print(
        f"80-D features : {audio_80.shape}"
    )

    print(
        f"240-D features: {audio_240.shape}"
    )

    print("\nClass distribution:")

    print(
        f"Label 0: {(y == 0).sum()}"
    )

    print(
        f"Label 1: {(y == 1).sum()}"
    )

    # --------------------------------------------------------
    # 1. ORIGINAL 40-D MFCC
    # --------------------------------------------------------

    result_40 = evaluate_representation(
        audio_40,
        y,
        "40-D MFCC MEAN",
    )

    # --------------------------------------------------------
    # 2. 80-D MFCC MEAN + STD
    # --------------------------------------------------------

    result_80 = evaluate_representation(
        audio_80,
        y,
        "80-D MFCC MEAN + STD",
    )

    # --------------------------------------------------------
    # 3. 240-D TEMPORAL REPRESENTATION
    # --------------------------------------------------------

    result_240 = evaluate_representation(
        audio_240,
        y,
        "240-D MFCC + DELTA + DELTA-DELTA "
        "(MEAN + STD)",
    )

    # --------------------------------------------------------
    # 4. FINAL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL AUDIO REPRESENTATION COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Representation':40s}"
        f"{'Accuracy':>18s}"
        f"{'Precision':>18s}"
        f"{'Recall':>18s}"
        f"{'F1':>18s}"
    )

    print("-" * 115)

    results = {
        "40-D MFCC Mean": result_40,
        "80-D MFCC Mean + Std": result_80,
        "240-D Temporal": result_240,
    }

    for name, result in results.items():

        accuracy = result["accuracy"]
        precision = result["precision"]
        recall = result["recall"]
        f1 = result["f1"]

        print(
            f"{name:40s}"
            f"{accuracy[0]:.4f} +/- {accuracy[1]:.4f}    "
            f"{precision[0]:.4f} +/- {precision[1]:.4f}    "
            f"{recall[0]:.4f} +/- {recall[1]:.4f}    "
            f"{f1[0]:.4f} +/- {f1[1]:.4f}"
        )

    # --------------------------------------------------------
    # F1 IMPROVEMENT
    # --------------------------------------------------------

    baseline_f1 = result_40["f1"][0]
    current_f1 = result_80["f1"][0]
    temporal_f1 = result_240["f1"][0]

    print("\nF1 changes:")

    print(
        f"80-D vs 40-D: "
        f"{current_f1 - baseline_f1:+.4f}"
    )

    print(
        f"240-D vs 80-D: "
        f"{temporal_f1 - current_f1:+.4f}"
    )

    print(
        f"240-D vs 40-D: "
        f"{temporal_f1 - baseline_f1:+.4f}"
    )

    print("\nIMPORTANT:")
    print("- Only the 83 training subjects were used.")
    print("- Validation data was NOT used.")
    print("- Test data was NOT used.")
    print("- Scaling was fitted independently inside each fold.")
    print("- No model was trained using validation/test data.")
    print("- The final test set remains locked.")


if __name__ == "__main__":
    main()