
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}


def extract_subject_features(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
) -> None:
    """
    Extract one MFCC feature vector per subject.

    Input:
        input_dir/
            S001.wav
            S002.wav
            ...

    Output:
        output_dir/
            S001.npy
            S002.npy
            ...

    Each output feature has shape:
        (40,)
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {input_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        file
        for file in input_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"No audio files found in: {input_path}")
        return

    for audio_file in tqdm(
        audio_files,
        desc="Extracting MFCC features",
    ):
        subject_id = audio_file.stem

        try:
            y, sr = librosa.load(
                str(audio_file),
                sr=sample_rate,
                mono=True,
            )

            if y.size == 0:
                print(
                    f"Skipping {subject_id}: "
                    "empty audio."
                )
                continue

            # 40 MFCC coefficients.
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=n_mfcc,
            )

            # Convert the time sequence into one
            # subject-level vector.
            subject_feature = np.mean(
                mfcc,
                axis=1,
            ).astype(np.float32)

            if subject_feature.shape != (n_mfcc,):
                print(
                    f"Skipping {subject_id}: "
                    f"unexpected shape "
                    f"{subject_feature.shape}"
                )
                continue

            output_file = output_path / f"{subject_id}.npy"
            np.save(output_file, subject_feature)

            print(
                f"{subject_id}: "
                f"shape={subject_feature.shape}"
            )

        except Exception as exc:
            print(
                f"Warning: failed to process "
                f"{audio_file.name}: {exc}"
            )

    print("Audio feature extraction completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract subject-level MFCC features."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing subject audio files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where .npy features will be saved.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    extract_subject_features(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
    )

