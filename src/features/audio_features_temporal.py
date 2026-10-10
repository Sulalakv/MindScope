from pathlib import Path
import argparse

import librosa
import numpy as np


def extract_features(
    audio_path,
    sample_rate=16000,
    n_mfcc=40,
):
    audio, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )

    delta = librosa.feature.delta(mfcc)

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2,
    )

    # --------------------------------------------------------
    # 1. MFCC mean
    # --------------------------------------------------------

    mfcc_mean = np.mean(
        mfcc,
        axis=1,
    )

    # --------------------------------------------------------
    # 2. MFCC standard deviation
    # --------------------------------------------------------

    mfcc_std = np.std(
        mfcc,
        axis=1,
    )

    # --------------------------------------------------------
    # 3. Delta mean
    # --------------------------------------------------------

    delta_mean = np.mean(
        delta,
        axis=1,
    )

    # --------------------------------------------------------
    # 4. Delta standard deviation
    # --------------------------------------------------------

    delta_std = np.std(
        delta,
        axis=1,
    )

    # --------------------------------------------------------
    # 5. Delta-delta mean
    # --------------------------------------------------------

    delta_delta_mean = np.mean(
        delta_delta,
        axis=1,
    )

    # --------------------------------------------------------
    # 6. Delta-delta standard deviation
    # --------------------------------------------------------

    delta_delta_std = np.std(
        delta_delta,
        axis=1,
    )

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            delta_mean,
            delta_std,
            delta_delta_mean,
            delta_delta_std,
        ]
    )

    return features.astype(
        np.float32
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
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

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_files = sorted(
        input_dir.glob("*.wav")
    )

    print(
        f"Found {len(audio_files)} audio files."
    )

    for audio_path in audio_files:

        subject_id = audio_path.stem

        try:

            features = extract_features(
                audio_path,
                sample_rate=args.sample_rate,
                n_mfcc=args.n_mfcc,
            )

            output_path = (
                output_dir /
                f"{subject_id}.npy"
            )

            np.save(
                output_path,
                features,
            )

            print(
                f"{subject_id}: "
                f"{features.shape}"
            )

        except Exception as e:

            print(
                f"ERROR - {subject_id}: {e}"
            )

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()