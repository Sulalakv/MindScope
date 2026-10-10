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
    Extract subject-level MFCC statistical features.

    For each subject:

        MFCC mean -> 40 values
        MFCC std  -> 40 values

        Final feature -> 80 values

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
        (80,)
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {input_path}"
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_files = sorted(
        file
        for file in input_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(
            f"No audio files found in: {input_path}"
        )
        return

    for audio_file in tqdm(
        audio_files,
        desc="Extracting MFCC statistical features",
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

            # ------------------------------------------------
            # MFCC
            # ------------------------------------------------

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=n_mfcc,
            )

            # ------------------------------------------------
            # Statistical representation
            # ------------------------------------------------

            mfcc_mean = np.mean(
                mfcc,
                axis=1,
            )

            mfcc_std = np.std(
                mfcc,
                axis=1,
            )

            # 40 mean + 40 standard deviation = 80
            subject_feature = np.concatenate(
                [
                    mfcc_mean,
                    mfcc_std,
                ]
            ).astype(np.float32)

            if subject_feature.shape != (
                n_mfcc * 2,
            ):
                print(
                    f"Skipping {subject_id}: "
                    f"unexpected shape "
                    f"{subject_feature.shape}"
                )
                continue

            output_file = (
                output_path /
                f"{subject_id}.npy"
            )

            np.save(
                output_file,
                subject_feature,
            )

            print(
                f"{subject_id}: "
                f"shape={subject_feature.shape}"
            )

        except Exception as exc:

            print(
                f"Warning: failed to process "
                f"{audio_file.name}: {exc}"
            )

    print(
        "Audio statistical feature extraction "
        "completed."
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract MFCC mean + standard "
            "deviation features."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing subject "
            "audio files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory where .npy features "
            "will be saved."
        ),
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