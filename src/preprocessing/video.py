
from pathlib import Path
import cv2


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def extract_frames(
    input_dir: str,
    output_dir: str,
    target_fps: int = 2,
) -> None:
    """
    Extract frames from videos at a fixed target FPS.

    Each video is treated as one subject.
    The video filename (without extension) becomes the subject ID.

    Example:
        S001.mp4
        -> output_dir/S001/frame_0000.jpg
        -> output_dir/S001/frame_0001.jpg
        -> ...
    """

    if target_fps <= 0:
        raise ValueError("target_fps must be greater than 0.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video directory not found: {input_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        file
        for file in input_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"No supported video files found in: {input_path}")
        return

    print(f"Found {len(video_files)} video(s).")

    for video_file in video_files:
        subject_id = video_file.stem
        subject_output_dir = output_path / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            print(f"Warning: Could not open {video_file.name}")
            continue

        original_fps = cap.get(cv2.CAP_PROP_FPS)

        if original_fps <= 0:
            print(f"Warning: Invalid FPS for {video_file.name}")
            cap.release()
            continue

        frame_interval = max(
            1,
            round(original_fps / target_fps)
        )

        frame_count = 0
        saved_count = 0

        print(f"Processing: {video_file.name}")

        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame_count % frame_interval == 0:
                frame_name = f"frame_{saved_count:04d}.jpg"
                frame_path = subject_output_dir / frame_name

                cv2.imwrite(str(frame_path), frame)
                saved_count += 1

            frame_count += 1

        cap.release()

        print(
            f"  Subject: {subject_id} | "
            f"Frames saved: {saved_count}"
        )

    print("Frame extraction completed.")


if __name__ == "__main__":
    extract_frames(
        input_dir="dataset/raw/videos",
        output_dir="dataset/processed/frames",
        target_fps=2,
    )

