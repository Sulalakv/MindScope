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


TRAIN_CSV = Path("../dataset/raw/PHQ/subjects.csv")
FACE_STATS_DIR = Path("../dataset/processed/face_features_stats")


def load_labels():
    df = pd.read_csv(TRAIN_CSV)

    labels = {}

    for _, row in df.iterrows():
        subject_id = str(row["subject_id"])
        labels[subject_id] = int(row["Label"])

    return labels


def load_features(subject_ids):
    features = []

    for subject_id in subject_ids:

        path = FACE_STATS_DIR / f"{subject_id}.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        feature = np.load(path)

        if feature.shape != (2560,):
            raise ValueError(
                f"{subject_id}: expected (2560,), "
                f"got {feature.shape}"
            )

        features.append(feature)

    return np.asarray(features, dtype=np.float32)


def evaluate(name, X, y, cv):

    scores = []

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        model = Pipeline([
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

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

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

        scores.append(f1)

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    print(
        f"\nMean F1: "
        f"{np.mean(scores):.4f} "
        f"+/- {np.std(scores):.4f}"
    )


def main():

    labels = load_labels()

    subject_ids = sorted(labels.keys())

    y = np.asarray(
        [labels[sid] for sid in subject_ids],
        dtype=np.int64,
    )

    features = load_features(subject_ids)

    print("Feature matrix:")
    print(features.shape)

    print("\nClass distribution:")
    print(f"Label 0: {np.sum(y == 0)}")
    print(f"Label 1: {np.sum(y == 1)}")

    # ---------------------------------------------------------
    # Split 2560-D representation
    # ---------------------------------------------------------

    face_mean = features[:, :1280]

    face_std = features[:, 1280:]

    print("\nRepresentation shapes:")
    print(f"Mean:       {face_mean.shape}")
    print(f"Std:        {face_std.shape}")
    print(f"Mean + Std: {features.shape}")

    # ---------------------------------------------------------
    # Feature statistics
    # ---------------------------------------------------------

    print("\nFeature statistics")
    print("-" * 70)

    print(
        f"Mean features:\n"
        f"  min  = {face_mean.min():.6f}\n"
        f"  max  = {face_mean.max():.6f}\n"
        f"  mean = {face_mean.mean():.6f}\n"
        f"  std  = {face_mean.std():.6f}"
    )

    print(
        f"\nStd features:\n"
        f"  min  = {face_std.min():.6f}\n"
        f"  max  = {face_std.max():.6f}\n"
        f"  mean = {face_std.mean():.6f}\n"
        f"  std  = {face_std.std():.6f}"
    )

    # ---------------------------------------------------------
    # Check whether any features are identical
    # ---------------------------------------------------------

    print("\nUnique subject feature vectors")
    print("-" * 70)

    print(
        "Mean vectors:",
        len(np.unique(face_mean, axis=0)),
        "/",
        len(subject_ids),
    )

    print(
        "Std vectors:",
        len(np.unique(face_std, axis=0)),
        "/",
        len(subject_ids),
    )

    # ---------------------------------------------------------
    # CV
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    evaluate(
        "FACE MEAN ONLY — 1280-D",
        face_mean,
        y,
        cv,
    )

    evaluate(
        "FACE STD ONLY — 1280-D",
        face_std,
        y,
        cv,
    )

    evaluate(
        "FACE MEAN + STD — 2560-D",
        features,
        y,
        cv,
    )


if __name__ == "__main__":
    main()