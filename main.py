# live_coach.py
# always start with source venv/Scripts/activate
import os
os.environ['GLOG_minloglevel'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle
import time
from collections import deque

from helper_functions import get_landmark_coordinates, angle, draw_debug, resize_window_to_screen
from punches import hands_up
from get_values import extract_features, get_stance_features, direction_facing
from model import PunchClassifier, StanceClassifier

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),
    (24,26),(26,28),(27,29),(29,31),(28,30),(30,32),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22)
]

# ── Load punch models — frontal AND sideways ────────────────
with open('label_encoder.pkl', 'rb') as f:
    le_frontal = pickle.load(f)
with open('label_encoder_sideways.pkl', 'rb') as f:
    le_sideways = pickle.load(f)

model_frontal = PunchClassifier(input_size=26, num_classes=len(le_frontal.classes_))
model_frontal.load_state_dict(torch.load('punch_classifier_best.pt'))
model_frontal.eval()
print(f"Frontal model loaded — classes: {list(le_frontal.classes_)}")

model_sideways = PunchClassifier(input_size=26, num_classes=len(le_sideways.classes_))
model_sideways.load_state_dict(torch.load('punch_classifier_sideways_best.pt'))
model_sideways.eval()
print(f"Sideways model loaded — classes: {list(le_sideways.classes_)}")

# ── Load stance model ──────────────────────────────────────
with open('stance_label_encoder.pkl', 'rb') as f:
    stance_le = pickle.load(f)

stance_num_classes = len(stance_le.classes_)
stance_model = StanceClassifier(input_size=10, num_classes=stance_num_classes)
stance_model.load_state_dict(torch.load('stance_classifier_best.pt'))
stance_model.eval()
print(f"Stance model loaded — classes: {list(stance_le.classes_)}")

# Smoothing — majority vote over last N stance predictions
STANCE_SMOOTH_WINDOW = 10
stance_history = deque(maxlen=STANCE_SMOOTH_WINDOW)

# ── Orientation tracking (drives which punch model runs) ────
DIRECTION_SMOOTH_WINDOW = 10
direction_history = deque(maxlen=DIRECTION_SMOOTH_WINDOW)
current_direction = 2          # 0=right facing, 1=left facing, 2/other=forward
current_sideways  = False

# ── Globals ────────────────────────────────────────────────
latest_result = None
frame_buffer  = []
timestamp_ms  = 0
frame_counter = 0

# Smoothing — majority vote over last N predictions, kept SEPARATE per
# orientation so a mid-flurry orientation flip doesn't blend two different
# label spaces into one vote
SMOOTH_WINDOW  = 5
CONF_THRESHOLD = 0.55
pred_history_frontal  = deque(maxlen=SMOOTH_WINDOW)
pred_history_sideways = deque(maxlen=SMOOTH_WINDOW)

# Punch display — hold label on screen for a moment after detection
displayed_label = None
displayed_conf  = 0.0
display_until   = 0  # timestamp to clear label

min_detection_confidence = 0.3
min_tracking_confidence = 0.6

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cap = cv2.VideoCapture(0)
window_width, window_height = resize_window_to_screen(cap)
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        cv2.namedWindow('Boxing Coach', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Boxing Coach', window_width, window_height)
        cv2.moveWindow('Boxing Coach', 0, 0)  # move to top-left corner of screen
        ret, frame = cap.read()
        if not ret:
            break

        # ── MediaPipe ──────────────────────────────────────
        camera_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        camera_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        inference_scale = 0.5  # scale down for inference
        inference_width = int(camera_width * inference_scale)
        inference_height = int(camera_height * inference_scale)

        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (inference_width, inference_height))
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_frame)
        timestamp_ms += 1
        #landmarker.detect_async(mp_image, timestamp_ms)
        frame_counter += 1
        if frame_counter % 2 == 0:
            landmarker.detect_async(mp_image, timestamp_ms)
        h, w, _ = frame.shape

        if latest_result and latest_result.pose_landmarks:
            landmarks = latest_result.pose_landmarks[0]

            # Draw skeleton
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
                    x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # ── Form feedback ──────────────────────────────

            guard_msg = hands_up(landmarks)
            guard_color = (0, 255, 0) if guard_msg == "good gaurd" else (0, 0, 255)
            draw_debug(frame, guard_msg, 3, guard_color)

            # ── Orientation detection — decides which punch model runs ──
            direction = direction_facing(landmarks)
            direction_history.append(direction)
            current_direction = max(set(direction_history), key=direction_history.count)
            current_sideways = current_direction in (0, 1)

        
            # ── Stance inference ────────────────────────────
            stance_features = get_stance_features(landmarks)
            stance_X = torch.tensor([stance_features], dtype=torch.float32)
            with torch.no_grad():
                stance_logits = stance_model(stance_X)
                stance_probs  = torch.softmax(stance_logits, dim=1)[0]
                stance_conf   = stance_probs.max().item()
                stance_pred   = stance_probs.argmax().item()
                stance_label  = stance_le.inverse_transform([stance_pred])[0]

            stance_history.append(stance_label)
            smoothed_stance = max(set(stance_history), key=stance_history.count)

            stance_color = (0, 255, 0) if smoothed_stance == 'orthodox' else (255, 0, 255)
            draw_debug(frame, f"Stance: {smoothed_stance} ({stance_conf:.0%})", 2, stance_color)

            # ── Feature extraction ─────────────────────────
            features = extract_features(landmarks)
            frame_buffer.append(features)
            if len(frame_buffer) > 30:
                frame_buffer.pop(0)

            # ── Punch model inference — swap model based on orientation ──
            if len(frame_buffer) == 30:
                if current_sideways:
                    active_model   = model_sideways
                    active_le      = le_sideways
                    active_history = pred_history_sideways
                else:
                    active_model   = model_frontal
                    active_le      = le_frontal
                    active_history = pred_history_frontal

                X = torch.tensor([frame_buffer], dtype=torch.float32)
                with torch.no_grad():
                    logits = active_model(X)
                    probs  = torch.softmax(logits, dim=1)[0]
                    conf   = probs.max().item()
                    pred   = probs.argmax().item()
                    label  = active_le.inverse_transform([pred])[0]

                # Only count confident predictions
                if conf >= CONF_THRESHOLD:
                    active_history.append(label)
                else:
                    active_history.append('none')

                # Majority vote over smoothing window
                if active_history:
                    smoothed = max(set(active_history), key=active_history.count)

                    # Hold non-none labels on screen for 1 second
                    if smoothed != 'none' and smoothed != 'guard':
                        displayed_label = smoothed
                        displayed_conf  = conf
                        display_until   = time.time() + 1.0

            # ── Draw punch label ───────────────────────────
            if displayed_label and time.time() < display_until:
                color = (0, 255, 255)
                draw_debug(frame, f"{displayed_label} ({displayed_conf:.0%})", 6, color)
            else:
                displayed_label = None

        cv2.putText(frame, "X = quit", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.imshow('Boxing Coach', frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

cap.release()
cv2.destroyAllWindows()