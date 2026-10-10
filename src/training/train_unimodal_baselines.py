from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dataset import MultimodalSubjectDataset


# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = "dataset/raw/PHQ/subjects.csv"
TRAIN_FACE_DIR = "dataset/processed/face_features"
TRAIN_AUDIO_DIR = "dataset/processed/audio_features"

VAL_CSV = "dataset_val/raw/PHQ/val.csv"
VAL_FACE_DIR = "dataset_val/features/face_features"
VAL_AUDIO_DIR = "dataset_val/features/audio_mfcc"

TEST_CSV = "dataset_test/raw/PHQ/test.csv"
TEST_FACE_DIR = "dataset_test/features/face_features"
TEST_AUDIO_DIR = "dataset_test/features/audio_mfcc"

OUTPUT_DIR = Path("models/unimodal_baselines")


# ============================================================
# LOAD DATA
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
# EVALUATION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(name, metrics):

    print("\n")
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {metrics['f1']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        metrics["confusion_matrix"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("UNIMODAL VS MULTIMODAL BASELINE EXPERIMENT")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading training data...")

    (
        X_train_face,
        X_train_audio,
        y_train,
        train_subjects
    ) = load_dataset(
        TRAIN_CSV,
        TRAIN_FACE_DIR,
        TRAIN_AUDIO_DIR
    )

    print("\nLoading validation data...")

    (
        X_val_face,
        X_val_audio,
        y_val,
        val_subjects
    ) = load_dataset(
        VAL_CSV,
        VAL_FACE_DIR,
        VAL_AUDIO_DIR
    )

    print("\nLoading test data...")

    (
        X_test_face,
        X_test_audio,
        y_test,
        test_subjects
    ) = load_dataset(
        TEST_CSV,
        TEST_FACE_DIR,
        TEST_AUDIO_DIR
    )

    # ========================================================
    # DATA INFORMATION
    # ========================================================

    print("\nDataset sizes:")

    print(
        f"Train      : {len(y_train)}"
    )

    print(
        f"Validation : {len(y_val)}"
    )

    print(
        f"Test       : {len(y_test)}"
    )

    print("\nFeature dimensions:")

    print(
        f"Face  : {X_train_face.shape[1]}"
    )

    print(
        f"Audio : {X_train_audio.shape[1]}"
    )

    # ========================================================
    # SUBJECT OVERLAP CHECK
    # ========================================================

    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    print("\nSubject overlap:")

    print(
        "Train ∩ Validation:",
        train_set.intersection(val_set)
    )

    print(
        "Train ∩ Test:",
        train_set.intersection(test_set)
    )

    print(
        "Validation ∩ Test:",
        val_set.intersection(test_set)
    )

    # ========================================================
    # DEFINE EXPERIMENTS
    # ========================================================

    experiments = {

        "Face Only": (
            X_train_face,
            X_val_face,
            X_test_face
        ),

        "Audio Only": (
            X_train_audio,
            X_val_audio,
            X_test_audio
        ),

        "Face + Audio": (
            np.concatenate(
                [X_train_face, X_train_audio],
                axis=1
            ),

            np.concatenate(
                [X_val_face, X_val_audio],
                axis=1
            ),

            np.concatenate(
                [X_test_face, X_test_audio],
                axis=1
            )
        )
    }

    # ========================================================
    # TRAIN + VALIDATE
    # ========================================================

    validation_results = {}

    trained_models = {}

    print("\n")
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for name, (
        X_train,
        X_val,
        X_test
    ) in experiments.items():

        print(f"\nTraining: {name}")

        model = create_model()

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_val,
            y_val
        )

        validation_results[name] = metrics

        trained_models[name] = model

        print_results(
            name,
            metrics
        )

        # Save model
        safe_name = (
            name.lower()
            .replace(" + ", "_")
            .replace(" ", "_")
        )

        model_path = (
            OUTPUT_DIR /
            f"{safe_name}.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"\nSaved: {model_path}"
        )

    # ========================================================
    # RANK BY VALIDATION F1
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VALIDATION RANKING")
    print("=" * 70)

    ranking = sorted(
        validation_results.items(),
        key=lambda item: item[1]["f1"],
        reverse=True
    )

    for rank, (
        name,
        metrics
    ) in enumerate(
        ranking,
        start=1
    ):

        print(
            f"{rank}. "
            f"{name:<18} "
            f"F1={metrics['f1']:.4f} "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

    # ========================================================
    # FINAL TEST COMPARISON
    #
    # We already know the experiment type before running the
    # test evaluation. Test remains completely untouched by
    # training and validation.
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL TEST COMPARISON")
    print("=" * 70)

    test_results = {}

    for name, model in trained_models.items():

        X_test = experiments[name][2]

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        test_results[name] = metrics

        print_results(
            name,
            metrics
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"{'Model':<20}"
        f"{'Val F1':<12}"
        f"{'Test F1':<12}"
        f"{'Test Acc':<12}"
    )

    print("-" * 70)

    for name in experiments:

        print(
            f"{name:<20}"
            f"{validation_results[name]['f1']:<12.4f}"
            f"{test_results[name]['f1']:<12.4f}"
            f"{test_results[name]['accuracy']:<12.4f}"
        )

    print("\nExperiment completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()