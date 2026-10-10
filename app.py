
from pathlib import Path
import secrets

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.inference.predictor import DepressionPredictor
from src.inference.pipeline import predict_from_video


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

ALLOWED_VIDEO = {".mp4", ".avi", ".mov"}
ALLOWED_AUDIO = {".wav", ".mp3", ".aac", ".ogg", ".m4a", ".mpeg"}

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static",
)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

predictor = DepressionPredictor()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    video = request.files.get("video")

    if not video or not video.filename:
        return render_template(
            "index.html",
            error="Please select a video containing speech.",
        ), 400

    video_ext = Path(secure_filename(video.filename)).suffix.lower()

    if video_ext not in ALLOWED_VIDEO:
        return render_template(
            "index.html",
            error="Unsupported video format. Use MP4, AVI, or MOV.",
        ), 400

    token = secrets.token_hex(12)
    video_path = UPLOAD_DIR / f"{token}_video{video_ext}"

    try:
        video.save(video_path)

        # The pipeline will extract speech from the video.
        result = predict_from_video(video_path, predictor)

        

        return render_template(
            "result.html",
            result=result,
        )

    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return render_template(
            "index.html",
            error=str(exc),
        ), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Processing failed. Please try another video.",
        ), 500

    finally:
        video_path.unlink(missing_ok=True)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error="The combined upload is too large. Maximum request size is 100 MB.",
    ), 413


if __name__ == "__main__":
    app.run(debug=False)
