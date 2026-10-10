
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_feature_extractor(device: torch.device):
    """
    Load ImageNet-pretrained MobileNetV2 and remove its classifier.

    Output feature size: 1280
    """

    weights = models.MobileNet_V2_Weights.DEFAULT

    model = models.mobilenet_v2(weights=weights)

    # Remove the final classification layer.
    model.classifier = nn.Identity()

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, weights.transforms()


def extract_subject_features(
    input_dir: str,
    output_dir: str,
    device: str | None = None,
) -> None:
    """
    Extract one 1280-dimensional face feature vector per subject.

    Input:
        input_dir/
            S001/
                frame_0000.jpg
                frame_0001.jpg
                ...

    Output:
        output_dir/
            S001.npy
            S002.npy
            ...
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Face directory not found: {input_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    print(f"Using device: {device}")

    model, preprocess = build_feature_extractor(device)

    subjects = sorted(
        folder
        for folder in input_path.iterdir()
        if folder.is_dir()
    )

    if not subjects:
        print(f"No subject folders found in: {input_path}")
        return

    for subject_dir in subjects:
        subject_id = subject_dir.name

        image_files = sorted(
            image
            for image in subject_dir.iterdir()
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            print(
                f"Skipping {subject_id}: "
                "no face images found."
            )
            continue

        frame_features = []

        for image_path in tqdm(
            image_files,
            desc=f"Face: {subject_id}",
        ):
            try:
                image = Image.open(image_path).convert("RGB")
                tensor = preprocess(image)
                tensor = tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    features = model(tensor)

                features = features.squeeze(0).cpu().numpy()
                frame_features.append(features)

            except Exception as exc:
                print(
                    f"Warning: failed to process "
                    f"{image_path}: {exc}"
                )

        if not frame_features:
            print(
                f"Skipping {subject_id}: "
                "no valid features extracted."
            )
            continue

        # Mean pooling converts frame-level features
        # into one subject-level representation.
        subject_feature = np.mean(
            np.stack(frame_features),
            axis=0,
        )

        output_file = output_path / f"{subject_id}.npy"
        np.save(output_file, subject_feature)

        print(
            f"Saved {subject_id}: "
            f"shape={subject_feature.shape}"
        )

    print("Face feature extraction completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract pretrained MobileNetV2 face features."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing subject folders with face images."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory where subject feature .npy files will be saved."
    )

    args = parser.parse_args()

    extract_subject_features(
        input_dir=args.input,
        output_dir=args.output,
    )

