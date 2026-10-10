
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "fusion_40d"


class FusionModel(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()

        self.face_branch = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.audio_branch = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, face, audio):
        face_features = self.face_branch(face)
        audio_features = self.audio_branch(audio)
        fused = torch.cat(
            [face_features, audio_features], dim=1
        )
        return self.classifier(fused)


class DepressionPredictor:
    def __init__(self):
        self.device = torch.device("cpu")

        self.face_scaler = joblib.load(
            MODEL_DIR / "face_scaler.joblib"
        )
        self.audio_scaler = joblib.load(
            MODEL_DIR / "audio_scaler.joblib"
        )

        self.model = FusionModel().to(self.device)

        checkpoint = torch.load(
            MODEL_DIR / "best_fusion_model.pth",
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval()

    def predict(self, face_features, audio_features):
        face = np.asarray(
            face_features, dtype=np.float32
        ).reshape(1, -1)

        audio = np.asarray(
            audio_features, dtype=np.float32
        ).reshape(1, -1)

        if face.shape[1] != 1280:
            raise ValueError(
                "Expected 1280 facial features."
            )

        if audio.shape[1] != 40:
            raise ValueError(
                "Expected 40 audio features."
            )

        if not (
            np.isfinite(face).all()
            and np.isfinite(audio).all()
        ):
            raise ValueError(
                "Features contain invalid numeric values."
            )

        face = self.face_scaler.transform(face)
        audio = self.audio_scaler.transform(audio)

        face_tensor = torch.tensor(
            face, dtype=torch.float32
        )
        audio_tensor = torch.tensor(
            audio, dtype=torch.float32
        )

        with torch.no_grad():
            logits = self.model(
                face_tensor, audio_tensor
            )
            probabilities = torch.softmax(
                logits, dim=1
            )[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        return {
            "prediction": predicted_class,
            "label": (
                "Depressed"
                if predicted_class == 1
                else "Non-Depressed"
            ),
            "class_scores": {
                "non_depressed": float(probabilities[0]),
                "depressed": float(probabilities[1]),
            },
        }
