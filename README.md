# Media Pipe Machine Learning Boxing Coach

A real-time computer vision system that watches you box through a webcam, figures out your stance and what punch you're throwing, and gives you form feedback. It's built as a full pipeline: pose extraction → feature engineering → data collection → model training → smoothed live inference.

## What the project does

1. Reads webcam frames and runs them through MediaPipe's pose landmarker to get 33 body keypoints per frame.
2. Converts those ra landmarks into scale and orientation based features vector (joint angles, wrist/elbow height compared to shoulders, hip rotation, etc.).
3. Buffers 30 frames of features into a short sequence and feeds it to an LSTM classifier to recognize the punch being thrown.
4. Simultaneously runs a separate classifier to detect stance (orthodox vs. southpaw).
5. Smooths motion with rolling-history buffers so a single noisy frame doesn't cause a flickery, wrong label.
6. Draws the skeleton, guard feedback, and predicted punch/stance back onto the live video feed.

## Project structure

```
├── collect_data.py        # Labeled training data collection via webcam
├── get_values.py          # Feature extraction (extract_features, get_stance, get_stance_features)
├── get_z.py                # Custom z-depth reconstruction (camera geometry + bone-length constraints)
├── helper_functions.py    # Shared utilities (angle calc, landmark helpers, cv_print functions)
├── punches.py               # Rule-based guard/stance/punch heuristics
├── model.py                 # PunchClassifier LSTM architecture
├── train.py                  # Offline training, evaluation, permutation/SHAP feature analysis
├── train_stances.py         # Offline training for the stance MLP
├── main.py                    #  Final product, the live coaching app
├── requirements.txt           #required packahes
├── analysis.py      #generates pytorch and SHAP analysis of feature weight, confusion matrixes, feature correlation, and training_curves
└── .gitignore
```

## Requirements

- **Python 3.10** ( MediaPipe's Tasks API has compatibility constraints on newer versions)
- A webcam 
- Windows / Mac / Linux (developed on Windows with Git Bash)

## Setup Instructions

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

The pose landmarker model file (`pose_landmarker_full.task`) is not included in this repo because it is too big. Download it from Google's MediaPipe model zoo and place it in the project root:

```bash
curl -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Usage Instructions

### Run the live coach

```bash
python main.py
```

Press **X** to quit.

### Collect your own training data

```bash
python collect_data.py
```
The training data original (`training_data.json`) I used to train the pickle and pytorch models  (e.g. `punch_classifier_best.pt` + `label_encoder.pkl`) is **not included** in this repo due to its size. Pretrained models are committed as paired release artifacts so `main.py` runs out of the box. To train your own model, run `collect_data.py` to record your own labeled reps. 

Controls: `q/w/e/r/t/y` label jab / cross / right hook / left hook / right uppercut / left uppercut; `0/1` label stance orthodox/southpaw; `S` saves the current buffered rep; `P` prints current feature values for debugging; `X` quits and saves.
Quitting and saving will create a `training_data.json`, or add to it if a custom one already exists.

### Retraining the models

```bash
python train.py           # punch classifier(s)
python train_stances.py   # stance classifier
```

`train.py` trains on `training_data.json`, saves `punch_classifier_best.pt` + `label_encoder.pkl`, `analysis.py`  produces training curves, a confusion matrix, permutation importance, and SHAP attribution plots.


## Model architecture

### PunchClassifier (LSTM)

```
Input:  (batch, 30 frames, 26 features)
   → LSTM(input_size=26, hidden_size=64, batch_first=True)
   → take last timestep's hidden state
   → Dropout(0.3)
   → Linear(64 → 32) → ReLU → Dropout(0.3)
   → Linear(32 → num_classes)
```

- **Sequence length:** 30 frames (a rolling window, not a fixed clip — the buffer slides forward every frame).
- **Input size:** 26 engineered features per frame (see below).
- **Output classes:** 6 punch types — jab, cross, left hook, right hook, left uppercut, right uppercut.
- **Dropout placement note:** dropout is applied as explicit `nn.Dropout` layers *after* the LSTM and after the first FC layer, rather than passed into the `nn.LSTM(..., dropout=0.3)` constructor. PyTorch's LSTM dropout only has an effect between stacked layers, so passing it into a single-layer LSTM is a silent no-op (and throws a warning) — moving it outside the LSTM actually regularizes the network.
- **Two trained variants:** because a person facing the camera vs. turned sideways produces very different feature geometry, there are two separately trained checkpoints rather than one model trying to generalize across both:
  - `punch_classifier_best.pt` — frontal orientation
  - `punch_classifier_sideways_best.pt` — sideways orientation
  - The sideways model currently only reliably recognizes one class — it needs more sideways training reps before it's as capable as the frontal model.

### StanceClassifier (MLP)

- A separate, much smaller feed-forward network trained on its own feature set (`get_stance_features()`), classifying orthodox vs. southpaw.
- Trained independently via `train_stances.py`, saved to its own `.pkl`/`.pt` artifacts, and kept fully decoupled from the punch model — stance and punch are different signals, and collapsing them into one model would mean one bad frame corrupts both predictions.
---

## Feature engineering (`get_values.py`)

Each frame is converted into a 26-value feature vector. x/y/z landmark positions alone are to imprecise based on a users orientation or distance from the camera, so everything is normalized before it ever reaches the model:

| # | Feature | # | Feature |
|---|---------|---|---------|
| 0–1 | elbow angle (L/R) | 13–15 | hip / leg angles |
| 2–3 | shoulder angle (L/R) | 16–17 | wrist height above elbow |
| 4–5 | wrist height vs. shoulder | 18–19 | elbow height vs. shoulder |
| 6–7 | wrist horizontal extension | 20–21 | wrist lateral position |
| 8–9 | wrist forward (z) extension | 22–23 | elbow flare |
| 10 | shoulder tilt | 24–25 | wrist-to-foot z extension |
| 11 | hip tilt | | |
| 12 | hip angle | | |

Key normalization principles used throughout:

- **Scale invariance:** distances are divided by `shoulder_width`, `hip_width`, or `body_height` rather than used as raw pixel/normalized-coordinate differences, so a person's size or distance from the camera doesn't change the feature values.
- **Direction canonicalization:** signed positional diffs (like `foot_x_diff`) flip sign depending on which way the person is facing the camera. Wherever this matters, the raw diff is multiplied by a direction sign before it's used, so "left foot forward" means the same thing regardless of camera orientation.
- **MediaPipe's left/right is camera-perspective, not person-perspective:** landmark 11 ("left shoulder" in MediaPipe's naming) is the shoulder on the *camera's* left, which is the person's actual *right* shoulder when facing the camera. This project treats left/right as person-perspective throughout, which means every place a landmark index is used, the mapping has to be sanity-checked — this has been a recurring source of bugs.

---

## How we calculate our own depth (z) — `get_z.py`

MediaPipe's built-in `z` landmark value is relative and noisy — it's not a physical unit, and it drifts frame to frame. Rather than trust it directly for wrist depth, the project reconstructs an approximate real-world 3D position using camera geometry and known human body proportions:

1. **Calibrate once:** measure the person's real height (172.7 cm), the camera's height off the floor (87.6 cm), and the camera's effective focal length in pixels (429.2 px at the 432×368 processing resolution) — derived from the camera's field of view.
2. **Estimate distance to the person (similar triangles):** the person's *apparent* height in pixels (nose landmark to ankle midpoint) is compared against their known real-world height. This ratio is stable frame-to-frame — a person's height doesn't change — unlike a single joint's position, which is exactly the noisy thing we're trying to measure.
3. **Cast a camera ray through each joint's pixel position:** every landmark's (x, y) pixel location is converted into a 3D unit direction vector from the camera (`pixel_to_ray`).
4. **Walk the kinematic chain outward using known bone lengths:** starting from the shoulder (anchored at the person's estimated distance), the elbow's actual 3D position is solved as the intersection of the elbow's camera ray with a sphere of radius = upper-arm length (29 cm, from Drillis & Contini anthropometric body-proportion tables) centered on the shoulder. The wrist is then solved the same way relative to the elbow, using forearm length (24 cm).
5. **Disambiguate the ray/sphere intersection:** a ray through a sphere has two intersection points (near and far). The correct one is chosen by picking whichever is closer to the joint's position in the *previous* frame, so depth doesn't randomly flip between two valid geometric solutions frame to frame.

This gives a physically-grounded depth estimate for the wrist even though the camera itself has no depth sensor — at the cost of assuming fixed calibration measurements (height, camera position) that would need to be redone per-user/per-setup for full accuracy. MediaPipe z-spread *between* landmarks is generally more reliable than trusting any single landmark's absolute z value, which is part of why this reconstruction leans on relative geometry (rays, bone-length spheres) instead.

---

## Live inference pipeline (`main.py`)

- MediaPipe runs in `LIVE_STREAM` async mode — frames are sent to the pose landmarker without blocking the main loop, and results arrive later via a callback.
- Features are buffered into a 30-frame rolling window; predictions only start once the buffer is full.
- **Confidence thresholding:** a prediction below `CONF_THRESHOLD` (0.55) is treated as `'none'` rather than trusted outright.
- **Temporal smoothing:** rather than acting on a single frame's prediction, a `deque`-based majority vote (`pred_history`, window of 5) is taken across recent predictions before a label is shown, which prevents one noisy frame from flashing a wrong punch label.
- **Display hold:** once a punch is confidently smoothed-detected, its label is held on screen for 1 second so fast punches are still readable even though the underlying prediction may have already moved on.
- Basic guard/form feedback (hands-up detection) runs alongside the classifier, independent of the ML models — it's a simple rule-based heuristic, not learned.
- `direction_history` and `stance_history` deques smooth orientation and stance decisions the same way, so a single ambiguous frame doesn't misroute which model or which frame-buffer a pose gets sent to.

### Motion gating (idle-frame suppression)

Because the LSTM sees a constantly-sliding window, it will happily produce a confident-looking punch label even when the person is just standing still, if the noise in that window vaguely resembles a punch shape. To suppress this:

- A heuristic **motion gate** thresholds a chosen motion-sensitive feature index (`MOTION_FEATURE_IDX`) against dual thresholds — `FIRE_THRESHOLD` to start counting a rep as "in motion" and a lower `RESET_THRESHOLD` to fall back to idle — rather than a single cutoff, which avoids rapid on/off flicker right at the boundary.
- This is treated as a stopgap (**"Option A"**). The longer-term plan (**"Option C"**) is to replace the heuristic with a small trained binary detector (moving vs. idle) instead of a hand-picked threshold on a single feature — a threshold on one noisy signal is fragile, whereas a trained detector can weigh multiple signals together.
- A recurring lesson from tuning this: compound **AND**-conditions across multiple noisy MediaPipe-derived signals tend to suppress *valid* detections, because it only takes one of several noisy signals to fall outside its window by chance. Weighted blends of signals, or gates applied sequentially rather than all-at-once, have worked better than stacking strict conditions.

---

## Model interpretability tooling (`train.py`)

- **Permutation importance:** shuffles one feature channel across validation samples (breaking that feature's link to the label) and measures how much validation accuracy drops — repeated multiple times per feature and averaged, since the shuffle is random.
- **SHAP (gradient-based attribution):** uses `GradientExplainer` against a background reference sample to attribute each prediction back to individual input features, both overall and per-punch-class, giving a sense of *how much* a feature pushes the output, not just whether the model relies on it.
- **Feature correlation matrix:** a Pearson correlation heatmap across all 26 features (flattened across samples and timesteps) to catch redundant or highly-correlated engineered features.
- **Candidate-checkpoint promotion:** training saves the best checkpoint so far and only overwrites the "best" model file if validation accuracy actually improves — with early stopping (patience of 15 epochs) to avoid overfitting on a small hand-collected dataset.

---

## Programs & libraries

| Purpose | Library |
|---|---|
| Pose estimation | MediaPipe 0.10.35 (`pose_landmarker_full.task`) |
| Computer vision / camera I/O | OpenCV (`opencv-python` 4.13.0.92) |
| Neural networks | PyTorch 2.12.1 |
| Classical ML utilities (label encoding, train/val split, confusion matrix) | scikit-learn 1.7.2 |
| Numerics | NumPy 2.2.6 |
| Plotting (loss curves, confusion matrix, feature importance) | Matplotlib 3.10.9 |
| Model interpretability | SHAP 0.49.1 (`GradientExplainer`) |

**Hardware:** Logitech C310 webcam, frames processed at 432×368 for pose estimation (kept separate from the full display resolution).

**Environment:** Python 3.10, Git Bash on Windows, VS Code, project at `~/Coding_Projects/Boxing_CV`.

---

## Model performance

Current punch classifier validation accuracy: **~78–87%** depending on training run, across 6 punch classes.

Known limitations:
- `left_uppercut` is frequently confused with `left_hook` — biomechanically similar motions that are hard to tell apart from a single camera angle.
- Scattered misclassifications on `right_uppercut`.
- A single webcam means limited native depth information; the custom `get_z.py` reconstruction helps but relies on fixed calibration constants (see above).

---

## What's missing 

- **No trained motion detector yet** — the idle/moving gate is still a hand-tuned heuristic (Option A above), not a learned binary classifier (Option C). The heuristic works but is fragile to re-tune if camera setup or motion style changes.
- **Sideways punch data is thin** — the sideways model currently recognizes essentially one class reliably; it needs a lot more labeled sideways reps before it's on par with the frontal model.
- **Depth (z) requires manual calibration** — the custom z-depth reconstruction depends on hardcoded real-world height/camera-height/focal-length constants measured once for one setup; it isn't automatically calibrated per user or per camera placement.
- **Single camera, no true stereo depth** — even with the kinematic-chain z-reconstruction, this is an estimate built on anthropometric averages and MediaPipe's noisy landmarks, not a hardware depth sensor. This is the likely cause of the project's most persistent classification confusion: uppercuts vs. hooks look biomechanically similar from one camera angle.
n.

## Roadmap

- Empirically validate and tune `MOTION_FEATURE_IDX` blend weights with a structured logging/tap process before moving on to Option C.
- Replace the heuristic motion gate with a trained binary (moving vs. idle) detector.
- Collect substantially more sideways training data to bring the sideways model up to parity with the frontal one.
- Camera geometry auto-calibration so the z-depth reconstruction doesn't depend on hand-measured constants.


## License

See [LICENSE](LICENSE).