
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.inference.predictor import FusionModel


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent

MODEL_DIR = ROOT / "models" / "fusion_40d"
FACE_DIR = DATA_ROOT / "dataset_test" / "features" / "face_features"
AUDIO_DIR = DATA_ROOT / "dataset_test" / "features" / "audio_mfcc"
CSV_PATH = DATA_ROOT / "dataset_test" / "raw" / "PHQ" / "test.csv"


def load_feature(folder, subject_id):
    path = folder / f"{subject_id}.npy"

    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    return np.load(path).reshape(-1)


def main():
    df = pd.read_csv(CSV_PATH)
    df["subject_id"] = df["subject_id"].astype(str).str.strip()

    face_scaler = joblib.load(MODEL_DIR / "face_scaler.joblib")
    audio_scaler = joblib.load(MODEL_DIR / "audio_scaler.joblib")

    device = torch.device("cpu")
    model = FusionModel().to(device)

    checkpoint = torch.load(
        MODEL_DIR / "best_fusion_model.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual = []
    predicted = []
    rows = []

    for _, row in df.iterrows():
        subject_id = row["subject_id"]
        true_label = int(row["Label"])

        face = load_feature(FACE_DIR, subject_id)
        audio = load_feature(AUDIO_DIR, subject_id)

        if face.size != 1280:
            raise ValueError(
                f"{subject_id}: expected 1280 face features, got {face.size}"
            )
        if audio.size != 40:
            raise ValueError(
                f"{subject_id}: expected 40 audio features, got {audio.size}"
            )

        face = face_scaler.transform(face.reshape(1, -1))
        audio = audio_scaler.transform(audio.reshape(1, -1))

        face_tensor = torch.tensor(face, dtype=torch.float32, device=device)
        audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(face_tensor, audio_tensor)
            scores = torch.softmax(logits, dim=1)[0].numpy()

        pred_label = int(np.argmax(scores))

        actual.append(true_label)
        predicted.append(pred_label)

        rows.append({
            "subject_id": subject_id,
            "actual": true_label,
            "predicted": pred_label,
            "non_depressed_score": float(scores[0]),
            "depressed_score": float(scores[1]),
        })

    print("\n=== PER-SUBJECT RESULTS ===")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== CLASS COUNTS ===")
    print("Actual:", pd.Series(actual).value_counts().sort_index().to_dict())
    print("Predicted:", pd.Series(predicted).value_counts().sort_index().to_dict())

    print("\n=== CONFUSION MATRIX ===")
    print("Rows = actual; columns = predicted; order = [0, 1]")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    print("\n=== METRICS ===")
    print(f"Accuracy:  {accuracy_score(actual, predicted):.4f}")
    print(f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(actual, predicted, zero_division=0):.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Non-Depressed (0)", "Depressed (1)"],
        zero_division=0,
    ))

    output_path = ROOT / "test_predictions_40d.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
