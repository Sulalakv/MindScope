from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_feature_extractor(device):
    """
    Load the same ImageNet-pretrained MobileNetV2
    configuration used by face_features.py.

    Output feature size: 1280
    """

    weights = models.MobileNet_V2_Weights.DEFAULT

    model = models.mobilenet_v2(weights=weights)

    model.classifier = nn.Identity()

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, weights.transforms()


def extract_frame_embeddings(
    input_dir,
    output_dir,
    device=None,
):
    """
    Extract frame-level 1280-D MobileNetV2 embeddings.

    Input:
        input_dir/
            S101/
                frame_000001.jpg
                frame_000002.jpg
                ...

    Output:
        output_dir/
            S101/
                S101_frame_000001.npy
                S101_frame_000002.npy
                ...
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Face directory not found: {input_path}"
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    print(f"Using device: {device}")

    model, preprocess = build_feature_extractor(device)

    subjects = sorted(
        folder
        for folder in input_path.iterdir()
        if folder.is_dir()
    )

    if not subjects:
        print(
            f"No subject folders found in: {input_path}"
        )
        return

    for subject_dir in subjects:

        subject_id = subject_dir.name

        image_files = sorted(
            image
            for image in subject_dir.iterdir()
            if image.is_file()
            and image.suffix.lower()
            in IMAGE_EXTENSIONS
        )

        if not image_files:
            print(
                f"Skipping {subject_id}: "
                "no face images found."
            )
            continue

        subject_output = (
            output_path / subject_id
        )

        subject_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        successful = 0

        for image_path in tqdm(
            image_files,
            desc=f"Face embeddings: {subject_id}",
        ):

            output_file = (
                subject_output
                / f"{image_path.stem}.npy"
            )

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                tensor = preprocess(image)

                tensor = (
                    tensor
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.no_grad():
                    features = model(tensor)

                features = (
                    features
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                if features.shape != (1280,):
                    raise ValueError(
                        f"Expected (1280,), "
                        f"got {features.shape}"
                    )

                np.save(
                    output_file,
                    features,
                )

                successful += 1

            except Exception as exc:

                print(
                    f"Warning: failed to process "
                    f"{image_path.name}: {exc}"
                )

        print(
            f"{subject_id}: "
            f"{successful}/{len(image_files)} "
            "frame embeddings saved."
        )

    print(
        "\nFrame-level face embedding extraction "
        "completed."
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract frame-level "
            "MobileNetV2 face embeddings."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Directory containing subject "
            "folders with face images."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Directory where frame-level "
            "embeddings will be saved."
        ),
    )

    args = parser.parse_args()

    extract_frame_embeddings(
        input_dir=args.input,
        output_dir=args.output,
    )