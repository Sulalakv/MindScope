from pathlib import Path

import joblib
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
# CONFIGURATION
# ============================================================

TRAIN_CSV = "dataset/raw/PHQ/subjects.csv"

TRAIN_FACE_DIR = "dataset/processed/face_features"
TRAIN_AUDIO_DIR = "dataset/processed/audio_features"

RANDOM_STATE = 42
N_SPLITS = 5

OUTPUT_DIR = Path("models/weighted_late_fusion")


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(csv_path, face_dir, audio_dir):

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
# CREATE SVM
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
# MAIN EXPERIMENT
# ============================================================

def run_weighted_late_fusion():

    print("=" * 70)
    print("WEIGHTED LATE-FUSION CROSS-VALIDATION")
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
    print(
        f"Class distribution: "
        f"{np.bincount(labels)}"
    )

    if len(np.unique(subjects)) != len(subjects):
        raise ValueError(
            "Duplicate subject IDs detected."
        )

    print("\n✓ All subject IDs are unique.")

    # --------------------------------------------------------
    # Same folds as previous experiments
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Store results for every weight and fold
    all_results = []

    # --------------------------------------------------------
    # Weight combinations
    #
    # face_weight + audio_weight = 1
    # --------------------------------------------------------

    weights = np.arange(
        0.0,
        1.01,
        0.1,
    )

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(face, labels),
        start=1,
    ):

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        train_subjects = set(
            subjects[train_idx]
        )

        val_subjects = set(
            subjects[val_idx]
        )

        overlap = train_subjects.intersection(
            val_subjects
        )

        if overlap:
            raise ValueError(
                f"Subject leakage detected: {overlap}"
            )

        print(
            f"Training subjects   : {len(train_idx)}"
        )

        print(
            f"Validation subjects : {len(val_idx)}"
        )

        print(
            "Subject overlap     : None"
        )

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
        # Evaluate every weighting
        # ----------------------------------------------------

        for face_weight in weights:

            audio_weight = 1.0 - face_weight

            fused_probability = (
                face_weight * face_probability
                +
                audio_weight * audio_probability
            )

            fused_prediction = (
                fused_probability >= 0.5
            ).astype(int)

            metrics = calculate_metrics(
                labels[val_idx],
                fused_prediction,
            )

            all_results.append({
                "fold": fold,

                "face_weight": round(
                    float(face_weight),
                    1,
                ),

                "audio_weight": round(
                    float(audio_weight),
                    1,
                ),

                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            })

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    # ========================================================
    # SUMMARY BY WEIGHT
    # ========================================================

    summary = (
        results_df
        .groupby(
            [
                "face_weight",
                "audio_weight",
            ]
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),

            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),

            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),

            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )

    # ========================================================
    # PRINT ALL WEIGHTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("WEIGHTED LATE-FUSION RESULTS")
    print("=" * 70)

    print(
        f"{'Face':<8}"
        f"{'Audio':<8}"
        f"{'Accuracy':<20}"
        f"{'Precision':<20}"
        f"{'Recall':<20}"
        f"{'F1':<20}"
    )

    print("-" * 96)

    for _, row in summary.iterrows():

        print(
            f"{row['face_weight']:<8.1f}"
            f"{row['audio_weight']:<8.1f}"
            f"{row['accuracy_mean']:.4f}"
            f" +/- {row['accuracy_std']:.4f}"
            f"   "
            f"{row['precision_mean']:.4f}"
            f" +/- {row['precision_std']:.4f}"
            f"   "
            f"{row['recall_mean']:.4f}"
            f" +/- {row['recall_std']:.4f}"
            f"   "
            f"{row['f1_mean']:.4f}"
            f" +/- {row['f1_std']:.4f}"
        )

    # ========================================================
    # RANKING BY F1
    # ========================================================

    ranking = summary.sort_values(
        by="f1_mean",
        ascending=False,
    ).reset_index(drop=True)

    print("\n")
    print("=" * 70)
    print("RANKING BY MEAN F1")
    print("=" * 70)

    for index, row in ranking.iterrows():

        print(
            f"{index + 1}. "
            f"Face {row['face_weight']:.1f} "
            f"+ Audio {row['audio_weight']:.1f} "
            f"→ F1 = "
            f"{row['f1_mean']:.4f} "
            f"+/- {row['f1_std']:.4f}"
        )

    # ========================================================
    # BEST WEIGHT
    # ========================================================

    best = ranking.iloc[0]

    print("\n")
    print("=" * 70)
    print("BEST WEIGHTED LATE-FUSION CONFIGURATION")
    print("=" * 70)

    print(
        f"Face weight  : "
        f"{best['face_weight']:.1f}"
    )

    print(
        f"Audio weight : "
        f"{best['audio_weight']:.1f}"
    )

    print(
        f"Mean Accuracy: "
        f"{best['accuracy_mean']:.4f} "
        f"+/- {best['accuracy_std']:.4f}"
    )

    print(
        f"Mean F1      : "
        f"{best['f1_mean']:.4f} "
        f"+/- {best['f1_std']:.4f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "weighted_late_fusion_results.joblib"
    )

    joblib.dump(
        {
            "fold_results": results_df,
            "summary": summary,
            "ranking": ranking,
            "best_configuration": best.to_dict(),
        },
        output_path,
    )

    print("\nResults saved to:")
    print(output_path)

    print(
        "\nWeighted late-fusion experiment completed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_weighted_late_fusion()