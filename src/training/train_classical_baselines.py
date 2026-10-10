from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

OUTPUT_DIR = Path("models/classical_baselines")


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

    faces = np.asarray(faces, dtype=np.float32)
    audios = np.asarray(audios, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    return faces, audios, labels, subjects


# ============================================================
# METRICS
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
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("CLASSICAL BASELINE EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading training data...")

    X_train_face, X_train_audio, y_train, train_subjects = load_dataset(
        TRAIN_CSV,
        TRAIN_FACE_DIR,
        TRAIN_AUDIO_DIR,
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("\nLoading validation data...")

    X_val_face, X_val_audio, y_val, val_subjects = load_dataset(
        VAL_CSV,
        VAL_FACE_DIR,
        VAL_AUDIO_DIR,
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test data...")

    X_test_face, X_test_audio, y_test, test_subjects = load_dataset(
        TEST_CSV,
        TEST_FACE_DIR,
        TEST_AUDIO_DIR,
    )

    # --------------------------------------------------------
    # Combine face + audio
    # --------------------------------------------------------

    X_train = np.concatenate(
        [X_train_face, X_train_audio],
        axis=1
    )

    X_val = np.concatenate(
        [X_val_face, X_val_audio],
        axis=1
    )

    X_test = np.concatenate(
        [X_test_face, X_test_audio],
        axis=1
    )

    print("\nDataset sizes:")
    print(f"Train: {len(y_train)}")
    print(f"Validation: {len(y_val)}")
    print(f"Test: {len(y_test)}")

    print("\nFeature dimensions:")
    print(f"Face : {X_train_face.shape[1]}")
    print(f"Audio: {X_train_audio.shape[1]}")
    print(f"Fusion: {X_train.shape[1]}")

    print("\nClass distribution:")
    print(f"Train: {np.bincount(y_train)}")
    print(f"Validation: {np.bincount(y_val)}")
    print(f"Test: {np.bincount(y_test)}")

    # --------------------------------------------------------
    # Verify subject independence
    # --------------------------------------------------------

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
    # MODELS
    # ========================================================

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),

        "Linear SVM": Pipeline([
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
        ]),

        "RBF SVM": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    # ========================================================
    # TRAIN + VALIDATE
    # ========================================================

    validation_results = {}

    print("\n")
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for name, model in models.items():

        print(f"\nTraining: {name}")

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

        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        safe_name = (
            name.lower()
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
            f"Saved: {model_path}"
        )

    # ========================================================
    # RANK MODELS BY VALIDATION F1
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL RANKING")
    print("=" * 70)

    ranking = sorted(
        validation_results.items(),
        key=lambda item: item[1]["f1"],
        reverse=True
    )

    for rank, (name, metrics) in enumerate(
        ranking,
        start=1
    ):

        print(
            f"{rank}. {name:<22} "
            f"F1={metrics['f1']:.4f} "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

    # ========================================================
    # SELECT BEST MODEL USING VALIDATION F1
    # ========================================================

    best_name = ranking[0][0]

    best_model = models[best_name]

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(f"Model: {best_name}")

    print(
        f"Validation F1: "
        f"{validation_results[best_name]['f1']:.4f}"
    )

    # ========================================================
    # FINAL TEST
    #
    # IMPORTANT:
    # Test is evaluated only after model selection.
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL TEST RESULT")
    print("=" * 70)

    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{test_metrics['f1']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        test_metrics["confusion_matrix"]
    )

    print("\nExperiment completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()