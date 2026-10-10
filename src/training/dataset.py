
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultimodalSubjectDataset(Dataset):
    """
    Subject-level multimodal dataset.

    Each subject contributes:
        Face  -> 1280 features
        Audio -> 40 features
        Label -> 0 or 1

    The face and audio features are matched using subject_id.
    """

    def __init__(
        self,
        csv_path: str,
        face_dir: str,
        audio_dir: str,
    ):
        self.csv_path = Path(csv_path)
        self.face_dir = Path(face_dir)
        self.audio_dir = Path(audio_dir)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}"
            )

        if not self.face_dir.exists():
            raise FileNotFoundError(
                f"Face feature directory not found: {self.face_dir}"
            )

        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Audio feature directory not found: {self.audio_dir}"
            )

        self.labels = pd.read_csv(self.csv_path)

        required_columns = {"subject_id", "Label"}

        missing_columns = required_columns - set(
            self.labels.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {missing_columns}"
            )

        self.samples = []

        for _, row in self.labels.iterrows():
            subject_id = str(row["subject_id"])
            label = int(row["Label"])

            face_path = self.face_dir / f"{subject_id}.npy"
            audio_path = self.audio_dir / f"{subject_id}.npy"

            if not face_path.exists():
                print(
                    f"Warning: missing face features for "
                    f"{subject_id}. Skipping."
                )
                continue

            if not audio_path.exists():
                print(
                    f"Warning: missing audio features for "
                    f"{subject_id}. Skipping."
                )
                continue

            face_features = np.load(face_path)
            audio_features = np.load(audio_path)

            face_features = np.asarray(
                face_features,
                dtype=np.float32,
            ).reshape(-1)

            audio_features = np.asarray(
                audio_features,
                dtype=np.float32,
            ).reshape(-1)

            if face_features.shape != (1280,):
                raise ValueError(
                    f"{subject_id}: expected face shape "
                    f"(1280,), got {face_features.shape}"
                )

            if audio_features.shape != (40,):
                raise ValueError(
                    f"{subject_id}: expected audio shape "
                    f"(40,), got {audio_features.shape}"
                )

            self.samples.append(
                {
                    "subject_id": subject_id,
                    "face": face_features,
                    "audio": audio_features,
                    "label": label,
                }
            )

        if not self.samples:
            raise ValueError(
                "No valid multimodal subjects were found."
            )

        print(
            f"Loaded {len(self.samples)} subjects "
            f"from {self.csv_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        face = torch.tensor(
            sample["face"],
            dtype=torch.float32,
        )

        audio = torch.tensor(
            sample["audio"],
            dtype=torch.float32,
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return {
            "subject_id": sample["subject_id"],
            "face": face,
            "audio": audio,
            "label": label,
        }

