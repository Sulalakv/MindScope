
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

import numpy as np

from src.features.audio_features import extract_subject_features as extract_audio_features
from src.features.face_features import extract_subject_features as extract_face_features
from src.preprocessing.audio import convert_to_wav
from src.preprocessing.face import extract_faces
from src.preprocessing.video import extract_frames
from src.inference.predictor import DepressionPredictor


SUPPORTED_VIDEO = {".mp4", ".avi", ".mov"}
SUPPORTED_AUDIO = {".wav", ".mp3", ".aac", ".ogg", ".m4a", ".mpeg"}


def predict_from_files(
    video_path: str | Path,
    audio_path: str | Path,
    predictor: DepressionPredictor,
) -> dict:
    """Extract features from uploaded files and run multimodal prediction."""

    video_path = Path(video_path)
    audio_path = Path(audio_path)

    if not video_path.is_file():
        raise FileNotFoundError("Video file was not found.")

    if not audio_path.is_file():
        raise FileNotFoundError("Audio file was not found.")

    if video_path.suffix.lower() not in SUPPORTED_VIDEO:
        raise ValueError("Unsupported video format.")

    if audio_path.suffix.lower() not in SUPPORTED_AUDIO:
        raise ValueError("Unsupported audio format.")

    with TemporaryDirectory(prefix="depression_ai_") as temp:
        root = Path(temp)

        video_input = root / "video_input"
        audio_input = root / "audio_input"
        frames_dir = root / "frames"
        faces_dir = root / "faces"
        face_features_dir = root / "face_features"
        wav_dir = root / "audio_wav"
        audio_features_dir = root / "audio_features"

        for folder in (
            video_input,
            audio_input,
            frames_dir,
            faces_dir,
            face_features_dir,
            wav_dir,
            audio_features_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        # Use the same subject ID for both modalities.
        subject_id = "uploaded_subject"

        video_copy = video_input / f"{subject_id}{video_path.suffix.lower()}"
        audio_copy = audio_input / f"{subject_id}{audio_path.suffix.lower()}"

        shutil.copy2(video_path, video_copy)
        shutil.copy2(audio_path, audio_copy)

        # Video -> frames -> detected face images -> 1280-D vector.
        extract_frames(
            input_dir=str(video_input),
            output_dir=str(frames_dir),
            target_fps=2,
        )

        extract_faces(
            input_dir=str(frames_dir),
            output_dir=str(faces_dir),
        )

        extract_face_features(
            input_dir=str(faces_dir),
            output_dir=str(face_features_dir),
            device="cpu",
        )

        face_file = face_features_dir / f"{subject_id}.npy"

        if not face_file.is_file():
            raise ValueError(
                "No usable facial features could be extracted. "
                "Try a clearer video with a visible face."
            )

        face_vector = np.load(face_file)

        # Audio -> mono 16 kHz WAV -> 40-D MFCC vector.
        convert_to_wav(
            input_dir=str(audio_input),
            output_dir=str(wav_dir),
        )

        wav_file = wav_dir / f"{subject_id}.wav"

        if not wav_file.is_file():
            raise ValueError(
                "Audio conversion failed. Please try another recording."
            )

        extract_audio_features(
            input_dir=str(wav_dir),
            output_dir=str(audio_features_dir),
            sample_rate=16000,
            n_mfcc=40,
        )

        audio_file = audio_features_dir / f"{subject_id}.npy"

        if not audio_file.is_file():
            raise ValueError(
                "Could not extract audio features from this recording."
            )

        audio_vector = np.load(audio_file)

        if face_vector.shape != (1280,):
            raise ValueError(
                f"Unexpected facial feature shape: {face_vector.shape}"
            )

        if audio_vector.shape != (40,):
            raise ValueError(
                f"Unexpected audio feature shape: {audio_vector.shape}"
            )

        # Both vectors are passed to the trained fusion model.
        return predictor.predict(face_vector, audio_vector)


    
def predict_from_video(
    video_path: str | Path,
    predictor: DepressionPredictor,
) -> dict:
    """Extract faces and speech from one video and predict."""

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError("Video file was not found.")

    with TemporaryDirectory(prefix="mindscope_") as temp:
        root = Path(temp)

        # Extract frames and facial features.
        video_input = root / "video"
        frames_dir = root / "frames"
        faces_dir = root / "faces"
        face_features_dir = root / "face_features"

        video_input.mkdir()
        subject_id = "uploaded_subject"

        video_copy = (
            video_input / f"{subject_id}{video_path.suffix.lower()}"
        )
        shutil.copy2(video_path, video_copy)

        extract_frames(
            input_dir=str(video_input),
            output_dir=str(frames_dir),
            target_fps=2,
        )

        extract_faces(
            input_dir=str(frames_dir),
            output_dir=str(faces_dir),
        )

        extract_face_features(
            input_dir=str(faces_dir),
            output_dir=str(face_features_dir),
            device="cpu",
        )

        face_file = face_features_dir / f"{subject_id}.npy"

        if not face_file.is_file():
            raise ValueError(
                "No usable face was detected. Try a clearer video."
            )

        face_vector = np.load(face_file)

        if face_vector.shape != (1280,):
            raise ValueError("Unexpected facial feature dimensions.")

        # Extract the video's audio track using FFmpeg.
        audio_dir = root / "audio"
        audio_dir.mkdir()

        wav_path = audio_dir / f"{subject_id}.wav"

        import subprocess

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-vn",
                    "-ac", "1",
                    "-ar", "16000",
                    "-y", str(wav_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "FFmpeg is not installed or is not available in PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                "Could not extract audio from this video. "
                "Check that the video contains an audio track."
            ) from exc

        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            raise ValueError("The video contains no usable audio.")

        audio_features_dir = root / "audio_features"
        audio_features_dir.mkdir()

        extract_audio_features(
            input_dir=str(audio_dir),
            output_dir=str(audio_features_dir),
            sample_rate=16000,
            n_mfcc=40,
        )

        audio_file = audio_features_dir / f"{subject_id}.npy"

        if not audio_file.is_file():
            raise ValueError("Could not extract speech features.")

        audio_vector = np.load(audio_file)

        if audio_vector.shape != (40,):
            raise ValueError("Unexpected audio feature dimensions.")

        return predictor.predict(face_vector, audio_vector)

