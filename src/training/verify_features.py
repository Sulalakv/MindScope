
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = Path("dataset/raw/PHQ/subjects.csv")
TRAIN_FACE_DIR = Path("dataset/processed/face_features")
TRAIN_AUDIO_DIR = Path("dataset/processed/audio_features")

VAL_CSV = Path("dataset_val/raw/PHQ/val.csv")
VAL_FACE_DIR = Path("dataset_val/features/face_agg")
VAL_AUDIO_DIR = Path("dataset_val/features/audio_mfcc")

TEST_CSV = Path("dataset_test/raw/PHQ/test.csv")
TEST_FACE_DIR = Path("dataset_test/features/face_agg")
TEST_AUDIO_DIR = Path("dataset_test/features/audio_mfcc")


# ============================================================
# LOAD CSV
# ============================================================

def load_split(name, csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{name} CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = {"subject_id", "Label"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} CSV is missing columns: {missing}"
        )

    df["subject_id"] = df["subject_id"].astype(str)

    return df


train_df = load_split("Training", TRAIN_CSV)
val_df = load_split("Validation", VAL_CSV)
test_df = load_split("Testing", TEST_CSV)


# ============================================================
# BASIC SPLIT INFORMATION
# ============================================================

print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(f"Training subjects   : {len(train_df)}")
print(f"Validation subjects : {len(val_df)}")
print(f"Testing subjects    : {len(test_df)}")
print(
    f"Total subjects      : "
    f"{len(train_df) + len(val_df) + len(test_df)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(name, df):

    counts = df["Label"].value_counts().sort_index()

    print(f"\n{name} class distribution:")

    for label in [0, 1]:
        print(
            f"  Label {label}: "
            f"{counts.get(label, 0)}"
        )


print_class_distribution("Training", train_df)
print_class_distribution("Validation", val_df)
print_class_distribution("Testing", test_df)


# ============================================================
# SUBJECT OVERLAP
# ============================================================

train_subjects = set(train_df["subject_id"])
val_subjects = set(val_df["subject_id"])
test_subjects = set(test_df["subject_id"])


train_val = train_subjects & val_subjects
train_test = train_subjects & test_subjects
val_test = val_subjects & test_subjects


print("\n" + "=" * 60)
print("SUBJECT OVERLAP CHECK")
print("=" * 60)

print("Train ∩ Validation:", train_val)
print("Train ∩ Test      :", train_test)
print("Validation ∩ Test :", val_test)


if train_val or train_test or val_test:
    raise RuntimeError(
        "Subject overlap detected. "
        "Do not train until the split is corrected."
    )

print("✓ No subject overlap detected.")


# ============================================================
# FEATURE CHECK
# ============================================================

def check_features(
    split_name,
    df,
    face_dir,
    audio_dir,
):
    print("\n" + "=" * 60)
    print(f"{split_name.upper()} FEATURE CHECK")
    print("=" * 60)

    missing_face = []
    missing_audio = []
    wrong_face_shape = []
    wrong_audio_shape = []
    invalid_face = []
    invalid_audio = []

    for subject_id in df["subject_id"]:

        face_path = face_dir / f"{subject_id}.npy"
        audio_path = audio_dir / f"{subject_id}.npy"

        # ----------------------------------------------------
        # Face
        # ----------------------------------------------------

        if not face_path.exists():
            missing_face.append(subject_id)
        else:
            try:
                face = np.load(face_path)

                if face.shape != (1280,):
                    wrong_face_shape.append(
                        (subject_id, face.shape)
                    )

                if not np.isfinite(face).all():
                    invalid_face.append(subject_id)

            except Exception as exc:
                invalid_face.append(
                    f"{subject_id}: {exc}"
                )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        if not audio_path.exists():
            missing_audio.append(subject_id)
        else:
            try:
                audio = np.load(audio_path)

                if audio.shape != (40,):
                    wrong_audio_shape.append(
                        (subject_id, audio.shape)
                    )

                if not np.isfinite(audio).all():
                    invalid_audio.append(subject_id)

            except Exception as exc:
                invalid_audio.append(
                    f"{subject_id}: {exc}"
                )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(f"Missing face features : {len(missing_face)}")
    print(f"Missing audio features: {len(missing_audio)}")

    print(
        f"Wrong face shapes     : "
        f"{len(wrong_face_shape)}"
    )

    print(
        f"Wrong audio shapes    : "
        f"{len(wrong_audio_shape)}"
    )

    print(
        f"Invalid face features : "
        f"{len(invalid_face)}"
    )

    print(
        f"Invalid audio features: "
        f"{len(invalid_audio)}"
    )

    if missing_face:
        print("\nMissing face subjects:")
        print(missing_face)

    if missing_audio:
        print("\nMissing audio subjects:")
        print(missing_audio)

    if wrong_face_shape:
        print("\nWrong face shapes:")
        print(wrong_face_shape)

    if wrong_audio_shape:
        print("\nWrong audio shapes:")
        print(wrong_audio_shape)

    if invalid_face:
        print("\nInvalid face features:")
        print(invalid_face)

    if invalid_audio:
        print("\nInvalid audio features:")
        print(invalid_audio)

    return {
        "missing_face": missing_face,
        "missing_audio": missing_audio,
        "wrong_face_shape": wrong_face_shape,
        "wrong_audio_shape": wrong_audio_shape,
        "invalid_face": invalid_face,
        "invalid_audio": invalid_audio,
    }


train_result = check_features(
    "Training",
    train_df,
    TRAIN_FACE_DIR,
    TRAIN_AUDIO_DIR,
)

val_result = check_features(
    "Validation",
    val_df,
    VAL_FACE_DIR,
    VAL_AUDIO_DIR,
)

test_result = check_features(
    "Testing",
    test_df,
    TEST_FACE_DIR,
    TEST_AUDIO_DIR,
)


# ============================================================
# FINAL RESULT
# ============================================================

all_problems = (
    train_result["missing_face"]
    + train_result["missing_audio"]
    + train_result["wrong_face_shape"]
    + train_result["wrong_audio_shape"]
    + train_result["invalid_face"]
    + train_result["invalid_audio"]
    + val_result["missing_face"]
    + val_result["missing_audio"]
    + val_result["wrong_face_shape"]
    + val_result["wrong_audio_shape"]
    + val_result["invalid_face"]
    + val_result["invalid_audio"]
    + test_result["missing_face"]
    + test_result["missing_audio"]
    + test_result["wrong_face_shape"]
    + test_result["wrong_audio_shape"]
    + test_result["invalid_face"]
    + test_result["invalid_audio"]
)


print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

if all_problems:
    print("❌ Problems were found.")
    print("Do NOT train the model yet.")
else:
    print("✅ ALL FEATURE CHECKS PASSED.")
    print()
    print("Face features : 1280 dimensions")
    print("Audio features: 40 dimensions")
    print("Splits        : subject-independent")
    print()
    print("The baseline training experiment is ready.")

