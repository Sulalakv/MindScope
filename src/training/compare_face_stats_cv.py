from pathlib import Path
import argparse

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


def load_labels(csv_path):
    df = pd.read_csv(csv_path)

    labels = {}

    for _, row in df.iterrows():
        subject_id = str(row["subject_id"])
        label = int(row["Label"])
        labels[subject_id] = label

    return labels


def load_features(feature_dir, subject_ids):
    features = []
    valid_subjects = []

    for subject_id in subject_ids:
        feature_path = feature_dir / f"{subject_id}.npy"

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing feature file: {feature_path}"
            )

        feature = np.load(feature_path)

        features.append(feature)
        valid_subjects.append(subject_id)

    return np.asarray(features, dtype=np.float32), valid_subjects


def evaluate_cv(X, y, cv):
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y), start=1
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

        accuracy = accuracy_score(y_val, predictions)
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

        fold_results.append({
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        print(
            f"    Fold {fold}: "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    return pd.DataFrame(fold_results)


def summarize_results(name, feature_dim, results):
    return {
        "Model": name,
        "Features": feature_dim,
        "Accuracy": results["accuracy"].mean(),
        "Accuracy Std": results["accuracy"].std(ddof=0),
        "Precision": results["precision"].mean(),
        "Precision Std": results["precision"].std(ddof=0),
        "Recall": results["recall"].mean(),
        "Recall Std": results["recall"].std(ddof=0),
        "F1": results["f1"].mean(),
        "F1 Std": results["f1"].std(ddof=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare 1280-D and 2560-D face representations "
            "using 5-fold subject-level cross-validation."
        )
    )

    parser.add_argument(
        "--train-csv",
        default="../dataset/raw/PHQ/subjects.csv",
    )

    parser.add_argument(
        "--face-mean-dir",
        default="../dataset/processed/face_features",
    )

    parser.add_argument(
        "--face-stats-dir",
        default="../dataset/processed/face_features_stats",
    )

    parser.add_argument(
        "--audio-stats-dir",
        default="../dataset/processed/audio_mfcc_stats",
    )

    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    face_mean_dir = Path(args.face_mean_dir)
    face_stats_dir = Path(args.face_stats_dir)
    audio_stats_dir = Path(args.audio_stats_dir)

    print("=" * 70)
    print("FACE REPRESENTATION COMPARISON")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load labels
    # ---------------------------------------------------------

    labels_dict = load_labels(train_csv)

    subject_ids = sorted(labels_dict.keys())

    y = np.asarray(
        [labels_dict[sid] for sid in subject_ids],
        dtype=np.int64,
    )

    print(f"\nTraining subjects: {len(subject_ids)}")
    print(f"Class 0: {np.sum(y == 0)}")
    print(f"Class 1: {np.sum(y == 1)}")

    # ---------------------------------------------------------
    # Load feature representations
    # ---------------------------------------------------------

    print("\nLoading face mean features...")
    face_mean, _ = load_features(
        face_mean_dir,
        subject_ids,
    )

    print(f"Face mean shape: {face_mean.shape}")

    print("\nLoading face mean + std features...")
    face_stats, _ = load_features(
        face_stats_dir,
        subject_ids,
    )

    print(f"Face mean + std shape: {face_stats.shape}")

    print("\nLoading audio mean + std features...")
    audio_stats, _ = load_features(
        audio_stats_dir,
        subject_ids,
    )

    print(f"Audio mean + std shape: {audio_stats.shape}")

    # ---------------------------------------------------------
    # Verify dimensions
    # ---------------------------------------------------------

    if face_mean.shape[1] != 1280:
        raise ValueError(
            f"Expected face mean dimension 1280, "
            f"got {face_mean.shape[1]}"
        )

    if face_stats.shape[1] != 2560:
        raise ValueError(
            f"Expected face stats dimension 2560, "
            f"got {face_stats.shape[1]}"
        )

    if audio_stats.shape[1] != 80:
        raise ValueError(
            f"Expected audio stats dimension 80, "
            f"got {audio_stats.shape[1]}"
        )

    # ---------------------------------------------------------
    # Create multimodal representations
    # ---------------------------------------------------------

    face_mean_audio = np.concatenate(
        [face_mean, audio_stats],
        axis=1,
    )

    face_stats_audio = np.concatenate(
        [face_stats, audio_stats],
        axis=1,
    )

    print("\nCombined feature dimensions:")
    print(
        f"Face 1280 + Audio 80  : "
        f"{face_mean_audio.shape[1]}"
    )
    print(
        f"Face 2560 + Audio 80  : "
        f"{face_stats_audio.shape[1]}"
    )

    # ---------------------------------------------------------
    # 5-fold subject-level CV
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    experiments = [
        (
            "Face Only (Mean)",
            1280,
            face_mean,
        ),
        (
            "Face Only (Mean + Std)",
            2560,
            face_stats,
        ),
        (
            "Audio Only (Mean + Std)",
            80,
            audio_stats,
        ),
        (
            "Face 1280 + Audio 80",
            1360,
            face_mean_audio,
        ),
        (
            "Face 2560 + Audio 80",
            2640,
            face_stats_audio,
        ),
    ]

    summary_results = []

    for name, dimension, X in experiments:

        print("\n" + "=" * 70)
        print(name)
        print(f"Feature dimension: {dimension}")
        print("=" * 70)

        results = evaluate_cv(
            X,
            y,
            cv,
        )

        summary = summarize_results(
            name,
            dimension,
            results,
        )

        summary_results.append(summary)

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------

    summary_df = pd.DataFrame(summary_results)

    print("\n")
    print("=" * 100)
    print("FINAL 5-FOLD CROSS-VALIDATION COMPARISON")
    print("=" * 100)

    for _, row in summary_df.iterrows():

        print(
            f"\n{row['Model']}"
        )

        print(
            f"  Accuracy : "
            f"{row['Accuracy']:.4f} "
            f"+/- {row['Accuracy Std']:.4f}"
        )

        print(
            f"  Precision: "
            f"{row['Precision']:.4f} "
            f"+/- {row['Precision Std']:.4f}"
        )

        print(
            f"  Recall   : "
            f"{row['Recall']:.4f} "
            f"+/- {row['Recall Std']:.4f}"
        )

        print(
            f"  F1       : "
            f"{row['F1']:.4f} "
            f"+/- {row['F1 Std']:.4f}"
        )

    # ---------------------------------------------------------
    # Ranking by F1
    # ---------------------------------------------------------

    ranked = summary_df.sort_values(
        by="F1",
        ascending=False,
    )

    print("\n")
    print("=" * 100)
    print("RANKING BY MEAN F1")
    print("=" * 100)

    for rank, (_, row) in enumerate(
        ranked.iterrows(),
        start=1,
    ):
        print(
            f"{rank}. {row['Model']}: "
            f"F1 = {row['F1']:.4f} "
            f"+/- {row['F1 Std']:.4f}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_path = Path(
        "models/face_representation_cv_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to: {output_path}"
    )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()