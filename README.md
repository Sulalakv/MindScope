# MindScope - Multimodal Depression Research Prototype


MindScope is a research prototype that explores depression-related patterns using facial and speech features extracted from video. It combines computer vision, audio feature extraction, and deep learning in a Flask web application.

> **Disclaimer:** This is an experimental research project, not a medical device. It cannot diagnose depression or replace assessment by a qualified mental health professional.



## Features



\- Video upload through a web interface

\- Face detection using MTCNN and OpenCV

\- Facial feature extraction using PyTorch and torchvision

\- Audio feature extraction using librosa

\- Feature-level fusion using a trained neural network

\- Prediction results with supportive, non-diagnostic guidance

\- Flask-based web interface



## Technology Stack



Python, Flask, PyTorch, torchvision, TensorFlow, MTCNN, OpenCV, librosa, NumPy, pandas, scikit-learn, joblib, HTML, CSS, and JavaScript.



## Project Structure



```text

Depression-AI-system/

â”œâ”€â”€ app/

â”‚   â”œâ”€â”€ static/

â”‚   â””â”€â”€ templates/

â”œâ”€â”€ dataset/              # Private data; not published

â”œâ”€â”€ models/

â”‚   â””â”€â”€ fusion_40d/       # Active model artifacts

â”œâ”€â”€ src/

â”‚   â”œâ”€â”€ features/

â”‚   â”œâ”€â”€ inference/

â”‚   â”œâ”€â”€ preprocessing/

â”‚   â””â”€â”€ training/

â”œâ”€â”€ tests/

â”œâ”€â”€ uploads/              # Temporary uploads; not published

â”œâ”€â”€ app.py

â”œâ”€â”€ evaluate_saved_model.py

â”œâ”€â”€ requirements.txt

â””â”€â”€ .gitignore

```



## Installation



Requirements:

\- Python 3.10

\- FFmpeg installed and available on PATH

\- Required trained model files



Create and activate a virtual environment:



```powershell

py -3.10 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Verify FFmpeg:



```powershell

ffmpeg -version

```



## Run the Application



From the project root:



```powershell

python app.py

```



Open `http://127.0.0.1:5000` in your browser if Flask reports that address.



The application requires compatible model weights and scaler files in the locations expected by the inference code. The research dataset is not included.



## Research and Evaluation



**Research paper:** *Efficient Multimodal Modeling for Automated Depression Identification through Facial and Speech Analysis*



The study used a self-collected dataset of 111 participants: 83 training, 14 validation, and 14 test subjects.



### Published paper result



The accepted paper reported **70% accuracy and an F1 score of 0.67** for its reported experiment.



### Current 40-dimensional fusion app model



A separate evaluation on the 14-subject held-out test set produced:



| Metric | Result |

|---|---:|

| Accuracy | 71.43% |

| Precision (class 1) | 0.60 |

| Recall (class 1) | 0.60 |

| F1 score (class 1) | 0.60 |



These are separate results and should not be treated as interchangeable. The held-out test set is small, and these results do not establish clinical validity or generalizability.



## Limitations



\- Small dataset and held-out test set

\- Potential overfitting

\- Predictions may be affected by lighting, video quality, face visibility, and audio quality

\- Model outputs are not clinically validated or calibrated

\- Performance on new participants and recording conditions is not established



## Privacy and Responsible Use



\- Do not upload participant videos, audio recordings, identifying information, or subject-level exports to a public repository.

\- Keep datasets, uploaded recordings, and secrets out of version control.

\- Obtain appropriate consent and authorization before collecting or processing research data.

\- Do not use this prototype for medical or other high-impact decisions.

\- Consult a qualified mental health professional for personal health concerns.



## License



No license has been specified yet. Add an appropriate license only after confirming that you have the right to publish the code and model artifacts.







