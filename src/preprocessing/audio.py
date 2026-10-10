
from pathlib import Path
import subprocess


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".aac",
    ".ogg",
    ".m4a",
    ".mpeg",
    ".mp4",
}


def convert_to_wav(
    input_dir: str,
    output_dir: str,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """
    Convert supported audio/video files to mono 16 kHz WAV files.

    Parameters
    ----------
    input_dir:
        Directory containing the original audio/video files.

    output_dir:
        Directory where converted WAV files will be saved.

    ffmpeg_path:
        FFmpeg executable name or full path.
        Default assumes ffmpeg is available in PATH.
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input audio directory not found: {input_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(
        file
        for file in input_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )

    if not files:
        print(f"No supported audio files found in: {input_path}")
        return

    print(f"Found {len(files)} audio/video file(s).")

    for input_file in files:
        output_file = output_path / f"{input_file.stem}.wav"

        command = [
            ffmpeg_path,
            "-i",
            str(input_file),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-y",
            str(output_file),
        ]

        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            print(f"Converted: {input_file.name}")

        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg was not found. Install FFmpeg or provide "
                "its full path using ffmpeg_path."
            )

        except subprocess.CalledProcessError:
            print(
                f"Warning: Failed to convert "
                f"{input_file.name}"
            )

    print("Audio conversion completed.")


if __name__ == "__main__":
    convert_to_wav(
        input_dir="dataset/raw/voices",
        output_dir="dataset/processed/audio_wav",
    )

