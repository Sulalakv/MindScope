from pathlib import Path
import argparse
import numpy as np


def extract_subject_statistics(feature_dir):
    """
    Compute mean + standard deviation over frame-level
    1280-D face embeddings for one subject.

    Output:
        2560-D vector = [mean(1280), std(1280)]
    """

    feature_files = sorted(feature_dir.glob("*.npy"))

    if not feature_files:
        raise FileNotFoundError(
            f"No .npy feature files found in {feature_dir}"
        )

    frame_features = []

    for file_path in feature_files:
        feature = np.load(file_path)

        if feature.shape != (1280,):
            raise ValueError(
                f"{file_path.name}: expected (1280,), "
                f"got {feature.shape}"
            )

        frame_features.append(feature)

    frame_features = np.asarray(frame_features, dtype=np.float32)

    mean_features = np.mean(frame_features, axis=0)
    std_features = np.std(frame_features, axis=0)

    subject_features = np.concatenate(
        [mean_features, std_features]
    )

    return subject_features.astype(np.float32)


def main():

    parser = argparse.ArgumentParser(
        description="Create 2560-D statistical face features."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing subject folders with frame embeddings."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where subject-level features will be saved."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    subject_dirs = sorted(
        path for path in input_dir.iterdir()
        if path.is_dir()
    )

    print(f"Found {len(subject_dirs)} subjects.")

    successful = 0
    failed = 0

    for subject_dir in subject_dirs:

        subject_id = subject_dir.name

        try:

            features = extract_subject_statistics(
                subject_dir
            )

            output_path = (
                output_dir / f"{subject_id}.npy"
            )

            np.save(
                output_path,
                features
            )

            print(
                f"{subject_id}: "
                f"{features.shape}"
            )

            successful += 1

        except Exception as e:

            print(
                f"ERROR - {subject_id}: {e}"
            )

            failed += 1

    print("\nFace statistical feature extraction complete.")

    print(f"Successful: {successful}")
    print(f"Failed    : {failed}")


if __name__ == "__main__":
    main()