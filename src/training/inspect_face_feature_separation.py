from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_CSV = Path("../dataset/raw/PHQ/subjects.csv")
FACE_DIR = Path("../dataset/processed/face_features")


def main():

    df = pd.read_csv(TRAIN_CSV)

    features = []
    labels = []
    subjects = []

    for _, row in df.iterrows():

        subject_id = str(row["subject_id"])
        label = int(row["Label"])

        feature_path = FACE_DIR / f"{subject_id}.npy"

        if not feature_path.exists():
            raise FileNotFoundError(feature_path)

        feature = np.load(feature_path)

        if feature.shape != (1280,):
            raise ValueError(
                f"{subject_id}: expected (1280,), "
                f"got {feature.shape}"
            )

        features.append(feature)
        labels.append(label)
        subjects.append(subject_id)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print("=" * 70)
    print("FACE FEATURE CLASS SEPARATION")
    print("=" * 70)

    print(f"\nFeature matrix: {X.shape}")

    for label in [0, 1]:

        class_features = X[y == label]

        print("\n" + "-" * 70)
        print(f"LABEL {label}")
        print("-" * 70)

        print(
            f"Subjects: {len(class_features)}"
        )

        print(
            f"Overall minimum : "
            f"{class_features.min():.6f}"
        )

        print(
            f"Overall maximum : "
            f"{class_features.max():.6f}"
        )

        print(
            f"Overall mean    : "
            f"{class_features.mean():.6f}"
        )

        print(
            f"Overall std     : "
            f"{class_features.std():.6f}"
        )

    # ---------------------------------------------------------
    # Per-subject embedding norms
    # ---------------------------------------------------------

    norms = np.linalg.norm(X, axis=1)

    print("\n")
    print("=" * 70)
    print("PER-SUBJECT EMBEDDING NORMS")
    print("=" * 70)

    print(
        f"\nLabel 0 mean norm: "
        f"{norms[y == 0].mean():.6f}"
    )

    print(
        f"Label 0 std norm : "
        f"{norms[y == 0].std():.6f}"
    )

    print(
        f"Label 1 mean norm: "
        f"{norms[y == 1].mean():.6f}"
    )

    print(
        f"Label 1 std norm : "
        f"{norms[y == 1].std():.6f}"
    )

    # ---------------------------------------------------------
    # Number of active / near-zero features
    # ---------------------------------------------------------

    near_zero = np.sum(
        X < 1e-6,
        axis=1,
    )

    print("\n")
    print("=" * 70)
    print("NEAR-ZERO ACTIVATIONS")
    print("=" * 70)

    print(
        f"\nLabel 0 mean near-zero features: "
        f"{near_zero[y == 0].mean():.2f}"
    )

    print(
        f"Label 1 mean near-zero features: "
        f"{near_zero[y == 1].mean():.2f}"
    )

    # ---------------------------------------------------------
    # Subject-level summary
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SUBJECT SUMMARY")
    print("=" * 70)

    summary = []

    for subject, label, feature, norm, zero_count in zip(
        subjects,
        y,
        X,
        norms,
        near_zero,
    ):

        summary.append({
            "subject_id": subject,
            "label": label,
            "embedding_mean": feature.mean(),
            "embedding_std": feature.std(),
            "embedding_min": feature.min(),
            "embedding_max": feature.max(),
            "embedding_norm": norm,
            "near_zero_count": zero_count,
        })

    summary_df = pd.DataFrame(summary)

    print(
        summary_df.to_string(
            index=False
        )
    )

    output_path = Path(
        "models/face_feature_subject_summary.csv"
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
        f"\nSummary saved to: {output_path}"
    )


if __name__ == "__main__":
    main()