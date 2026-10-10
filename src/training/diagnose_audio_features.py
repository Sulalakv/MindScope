from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_CSV = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "PHQ"
    / "subjects.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "dataset_val"
    / "raw"
    / "PHQ"
    / "val.csv"
)

TRAIN_AUDIO_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "audio_features"
)

VAL_AUDIO_DIR = (
    PROJECT_ROOT
    / "dataset_val"
    / "features"
    / "audio_mfcc"
)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_audio_features(subject_ids, audio_dir):

    features = []
    valid_subjects = []

    for subject_id in subject_ids:

        path = audio_dir / f"{subject_id}.npy"

        if not path.exists():
            print(f"Missing: {subject_id}")
            continue

        feature = np.load(path).reshape(-1)

        features.append(feature)
        valid_subjects.append(subject_id)

    return np.asarray(features, dtype=np.float32), valid_subjects


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO FEATURE DIAGNOSTIC")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_subjects = train_df["subject_id"].astype(str).tolist()
    val_subjects = val_df["subject_id"].astype(str).tolist()

    train_labels = dict(
        zip(
            train_df["subject_id"].astype(str),
            train_df["Label"].astype(int),
        )
    )

    val_labels = dict(
        zip(
            val_df["subject_id"].astype(str),
            val_df["Label"].astype(int),
        )
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X_train, train_valid = load_audio_features(
        train_subjects,
        TRAIN_AUDIO_DIR,
    )

    X_val, val_valid = load_audio_features(
        val_subjects,
        VAL_AUDIO_DIR,
    )

    y_train = np.array(
        [train_labels[s] for s in train_valid]
    )

    y_val = np.array(
        [val_labels[s] for s in val_valid]
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\nFEATURE SHAPES")
    print("-" * 70)

    print("Training :", X_train.shape)
    print("Validation:", X_val.shape)

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    print("\nGLOBAL STATISTICS")
    print("-" * 70)

    print(f"Train minimum : {X_train.min():.6f}")
    print(f"Train maximum : {X_train.max():.6f}")
    print(f"Train mean    : {X_train.mean():.6f}")
    print(f"Train std     : {X_train.std():.6f}")

    print(f"\nVal minimum   : {X_val.min():.6f}")
    print(f"Val maximum   : {X_val.max():.6f}")
    print(f"Val mean      : {X_val.mean():.6f}")
    print(f"Val std       : {X_val.std():.6f}")

    # --------------------------------------------------------
    # Per-feature statistics
    # --------------------------------------------------------

    train_feature_std = X_train.std(axis=0)

    print("\nPER-FEATURE STANDARD DEVIATION")
    print("-" * 70)

    for i, std in enumerate(train_feature_std):

        print(
            f"MFCC {i + 1:02d}: "
            f"mean={X_train[:, i].mean():.6f}, "
            f"std={std:.6f}, "
            f"min={X_train[:, i].min():.6f}, "
            f"max={X_train[:, i].max():.6f}"
        )

    # --------------------------------------------------------
    # Check duplicate / nearly identical vectors
    # --------------------------------------------------------

    print("\nDUPLICATE / SIMILARITY CHECK")
    print("-" * 70)

    unique_train = np.unique(
        X_train,
        axis=0,
    ).shape[0]

    unique_val = np.unique(
        X_val,
        axis=0,
    ).shape[0]

    print(f"Unique training vectors   : {unique_train}/{len(X_train)}")
    print(f"Unique validation vectors : {unique_val}/{len(X_val)}")

    # --------------------------------------------------------
    # Subject-level values
    # --------------------------------------------------------

    print("\nTRAINING SUBJECT AUDIO FEATURES")
    print("-" * 70)

    for i, subject_id in enumerate(train_valid[:20]):

        print(
            f"{subject_id} | "
            f"Label={y_train[i]} | "
            f"Mean={X_train[i].mean():.6f} | "
            f"Std={X_train[i].std():.6f}"
        )

    # --------------------------------------------------------
    # Class-level statistics
    # --------------------------------------------------------

    print("\nCLASS-LEVEL STATISTICS")
    print("-" * 70)

    for label in [0, 1]:

        class_features = X_train[y_train == label]

        print(f"\nLabel {label}")
        print(f"Subjects : {len(class_features)}")
        print(f"Mean     : {class_features.mean():.6f}")
        print(f"Std      : {class_features.std():.6f}")

    # --------------------------------------------------------
    # Validation statistics
    # --------------------------------------------------------

    print("\nVALIDATION SUBJECT AUDIO FEATURES")
    print("-" * 70)

    for i, subject_id in enumerate(val_valid):

        print(
            f"{subject_id} | "
            f"Label={y_val[i]} | "
            f"Mean={X_val[i].mean():.6f} | "
            f"Std={X_val[i].std():.6f}"
        )

    print("\n" + "=" * 70)
    print("AUDIO FEATURE DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()