
from pathlib import Path

import cv2
from mtcnn import MTCNN
from tqdm import tqdm


def extract_faces(
    input_dir: str,
    output_dir: str,
    image_size: tuple[int, int] = (224, 224),
) -> None:
    """
    Detect and extract faces from subject-wise video frames.

    Only frames containing exactly one detected face are kept.

    Expected input:
        input_dir/
            S001/
                frame_0000.jpg
                frame_0001.jpg
                ...

    Output:
        output_dir/
            S001/
                frame_0000.jpg
                frame_0001.jpg
                ...
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input frame directory not found: {input_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    detector = MTCNN()

    subjects = sorted(
        folder
        for folder in input_path.iterdir()
        if folder.is_dir()
    )

    if not subjects:
        print(f"No subject folders found in: {input_path}")
        return

    total_input = 0
    total_saved = 0

    for subject_dir in subjects:
        subject_id = subject_dir.name
        save_dir = output_path / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            image
            for image in subject_dir.iterdir()
            if image.is_file()
            and image.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        print(f"\nProcessing subject: {subject_id}")

        subject_saved = 0

        for image_path in tqdm(images, desc=subject_id):
            total_input += 1

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            faces = detector.detect_faces(rgb_image)

            # Keep only frames with exactly one face.
            if len(faces) != 1:
                continue

            x, y, width, height = faces[0]["box"]

            # MTCNN can occasionally return negative coordinates.
            x = max(0, x)
            y = max(0, y)

            if width <= 0 or height <= 0:
                continue

            face = image[
                y:y + height,
                x:x + width,
            ]

            if face.size == 0:
                continue

            face = cv2.resize(
                face,
                image_size,
                interpolation=cv2.INTER_AREA,
            )

            output_file = save_dir / image_path.name

            success = cv2.imwrite(
                str(output_file),
                face,
            )

            if success:
                subject_saved += 1
                total_saved += 1

        print(
            f"  Input frames: {len(images)} | "
            f"Faces saved: {subject_saved}"
        )

    print("\nFace extraction completed.")
    print(f"Total input frames: {total_input}")
    print(f"Total faces saved: {total_saved}")


if __name__ == "__main__":
    extract_faces(
        input_dir="dataset/processed/frames",
        output_dir="dataset/processed/clean_faces",
    )

