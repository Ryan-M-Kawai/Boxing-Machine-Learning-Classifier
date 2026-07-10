# Boxing CV — Real-Time Punch Classification & Coaching

A real-time computer vision system that classifies boxing punches from live webcam input using MediaPipe pose estimation and an LSTM neural network. Built as an end-to-end pipeline: pose extraction → feature engineering → data collection → model training → live inference with coaching feedback.

## Demo

*(optional: add a GIF or screenshot of `main.py` running here once you have one)*

## Features

- Real-time pose estimation via MediaPipe Tasks API (`pose_landmarker_full.task`)
- 26-dimensional engineered feature vector per frame (joint angles, wrist/elbow height, lateral extension, hip rotation, etc.)
- LSTM-based sequence classifier over a 30-frame sliding window
- Six punch classes: jab, cross, left hook, right hook, left uppercut, right uppercut
- Live confidence smoothing via majority vote over recent predictions
- Basic guard/form feedback (hands-up detection, chin protection)
- Model interpretability tooling: permutation importance, SHAP gradient attribution, feature correlation analysis

## Project Structure

```
├── collect_data.py       # Labeled training data collection via webcam
├── get_values.py         # Feature extraction (extract_features, get_stance)
├── helper_functions.py   # Shared utilities (angle calc, landmark helpers)
├── punches.py             # Rule-based guard/stance/punch heuristics
├── model.py               # PunchClassifier LSTM architecture
├── train.py               # Offline training, evaluation, feature analysis
├── main.py                 # Live inference — the actual coaching app
├── check_data.py           # Inspect training_data.json class balance
├── reformat_data.py        # Pretty-print training_data.json
├── requirements.txt
└── .gitignore
```

## Requirements

- **Python 3.10** (required — MediaPipe's Tasks API has compatibility constraints on newer versions)
- A webcam
- Windows / Mac / Linux (developed on Windows with Git Bash)

## Setup

```bash
# Clone the repo
git clone https://github.com/Ryan-M-Kawai/Boxing-Machine-Learning-Classifier.git
cd Boxing-Machine-Learning-Classifier

# Create and activate a virtual environment (Python 3.10)
python3.10 -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# source venv/bin/activate        # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download the MediaPipe pose model

The pose landmarker model file (`pose_landmarker_full.task`) is not included in this repo (large third-party binary). Download it from Google's MediaPipe model zoo and place it in the project root:

```bash
curl -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

*(check the [MediaPipe pose landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) for the current download link if this changes)*

## Usage

### Run the live coach (pretrained model included)

```bash
python main.py
```

Press **Q** to quit.

### Collect your own training data

```bash
python collect_data.py
```

Controls:
- `j` / `c` / `h` / `k` / `u` / `y` — label as jab / cross / right hook / left hook / right uppercut / left uppercut
- `g` — guard, `n` — none
- `S` — save the current buffered rep
- `P` — print current feature values (debug)
- `Q` — quit and save

### Retrain the model

```bash
python train.py
```

Trains on `training_data.json` (not included — see below), saves `punch_classifier_best.pt` and `label_encoder.pkl`, and produces training curves, confusion matrix, permutation importance, and SHAP attribution plots.

## About the training data

Raw training data (`training_data.json`) is **not included** in this repo due to size and because it's personal recorded pose data. A pretrained model (`punch_classifier_best.pt` + `label_encoder.pkl`) is included so `main.py` runs out of the box. To train your own model, run `collect_data.py` to record your own labeled reps.

## Model performance

Current validation accuracy: **~78–87%** depending on training run, on 6 punch classes.

Known limitations:
- `left_uppercut` is frequently confused with `left_hook` — biomechanically similar motions from a front-facing single camera
- Scattered misclassifications on `right_uppercut`
- Single front-facing webcam means limited depth information; MediaPipe's z-axis is noisy, so most angle features are computed in 2D only

## Architecture notes

- **Feature extraction**: joint angles (elbow, shoulder, hip, leg) computed in 2D (z-depth excluded due to single-camera noise), plus normalized wrist/elbow height and lateral position relative to shoulder center.
- **Model**: single-layer LSTM (hidden size 64) → FC(32) → FC(num_classes), with dropout applied as explicit `nn.Dropout` layers after the LSTM and FC layers (not inside the LSTM constructor, to avoid PyTorch's single-layer dropout no-op warning).
- **Landmark convention**: MediaPipe's left/right labels are camera-perspective, not person-perspective. When facing the camera without a mirrored feed, MediaPipe's "left" landmark corresponds to the person's actual right side. This project treats "left/right" as person-perspective throughout — see comments in `get_values.py` / `collect_data.py` for where this is handled.

## Roadmap

- Stance classification as an auxiliary input feature (trained MLP predicting orthodox/southpaw, appended to the punch feature vector)
- Camera geometry calibration (focal length + camera height + person height) to derive real-world distance and pose orientation
- Improved uppercut/hook disambiguation, likely via additional camera angle or depth sensing

## License

See [LICENSE](LICENSE).
