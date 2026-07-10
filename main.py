# # live_coach.py
# # always start with source venv/Scripts/activate
# import os
# os.environ['GLOG_minloglevel'] = '3'

# import cv2
# import mediapipe as mp
# import numpy as np
# import torch
# import pickle
# import time
# from collections import deque

# from helper_functions import get_landmark_coordinates, angle, draw_debug
# from punches import hands_up
# from get_values import extract_features
# from model import PunchClassifier

# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

# POSE_CONNECTIONS = [
#     (11,12),(11,13),(13,15),(12,14),(14,16),
#     (11,23),(12,24),(23,24),(23,25),(25,27),
#     (24,26),(26,28),(27,29),(29,31),(28,30),(30,32),
#     (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
#     (9,10),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22)
# ]

# # ── Load model ─────────────────────────────────────────────
# with open('label_encoder.pkl', 'rb') as f:
#     le = pickle.load(f)

# num_classes = len(le.classes_)
# model = PunchClassifier(input_size=26, num_classes=num_classes)
# model.load_state_dict(torch.load('punch_classifier_best.pt'))
# model.eval()
# print(f"Model loaded — classes: {list(le.classes_)}")

# # ── Globals ────────────────────────────────────────────────
# latest_result = None
# latest_ts = 0
# frame_buffer  = []
# timestamp_ms  = 0
# frame_counter = 0
# # Smoothing — majority vote over last N predictions
# SMOOTH_WINDOW  = 5
# CONF_THRESHOLD = 0.55
# pred_history   = deque(maxlen=SMOOTH_WINDOW)  # stores (label, conf) tuples

# # Punch display — hold label on screen for a moment after detection
# displayed_label = None
# displayed_conf  = 0.0
# display_until   = 0  # timestamp to clear label

# min_detection_confidence = 0.3
# min_tracking_confidence = 0.6

# last_appended_ts = -1  # start at a value that can never match a real timestamp

# def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     global latest_result, latest_ts
#     latest_result = result
#     latest_ts = timestamp_ms

# options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)

# cap = cv2.VideoCapture(1)

# with PoseLandmarker.create_from_options(options) as landmarker:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # ── MediaPipe ──────────────────────────────────────
#         rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         small_frame = cv2.resize(rgb_frame, (432, 368))
#         mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_frame)
#         timestamp_ms += 1
#         frame_counter += 1
#         if frame_counter % 2 == 0:
#             landmarker.detect_async(mp_image, timestamp_ms)
#         h, w, _ = frame.shape

#         if latest_result and latest_result.pose_landmarks and latest_ts != last_appended_ts:
#             landmarks = latest_result.pose_landmarks[0]

#             # ── Feature extraction (only on genuinely new poses) ──
#             features = extract_features(landmarks)
#             frame_buffer.append(features)
#             if len(frame_buffer) > 30:
#                 frame_buffer.pop(0)
#             last_appended_ts = latest_ts

#             # Draw skeleton
#             for lm in landmarks:
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#             for start_idx, end_idx in POSE_CONNECTIONS:
#                 if start_idx < len(landmarks) and end_idx < len(landmarks):
#                     x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
#                     x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
#                     cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

#             # ── Form feedback ──────────────────────────────
#             guard_msg = hands_up(landmarks)
#             guard_color = (0, 255, 0) if guard_msg == "good gaurd" else (0, 0, 255)
#             draw_debug(frame, guard_msg, 3, guard_color)

#             # ── Model inference ────────────────────────────
#             if len(frame_buffer) == 30:
#                 X = torch.tensor([frame_buffer], dtype=torch.float32)
#                 with torch.no_grad():
#                     logits = model(X)
#                     probs  = torch.softmax(logits, dim=1)[0]
#                     conf   = probs.max().item()
#                     pred   = probs.argmax().item()
#                     label  = le.inverse_transform([pred])[0]

#                 # Only count confident predictions
#                 if conf >= CONF_THRESHOLD:
#                     pred_history.append((label, conf))
#                 else:
#                     pred_history.append(('none', conf))

#                 # Majority vote over smoothing window
#                 if pred_history:
#                     labels_only = [l for l, c in pred_history]
#                     smoothed = max(set(labels_only), key=labels_only.count)
#                     matching_confs = [c for l, c in pred_history if l == smoothed]
#                     smoothed_conf = sum(matching_confs) / len(matching_confs)

#                     # Hold non-none labels on screen for 1 second
#                     if smoothed != 'none' and smoothed != 'guard':
#                         displayed_label = smoothed
#                         displayed_conf  = smoothed_conf
#                         display_until   = time.time() + 1.0

#             # ── Draw punch label ───────────────────────────
#             if displayed_label and time.time() < display_until:
#                 color = (0, 255, 255)
#                 draw_debug(frame, f"{displayed_label} ({displayed_conf:.0%})", 6, color)
#             else:
#                 displayed_label = None

#         cv2.putText(frame, "Q = quit", (10, h - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
#         cv2.imshow('Boxing Coach', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()
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

from helper_functions import get_landmark_coordinates, angle, draw_debug
from punches import hands_up
from get_values import extract_features
from model import PunchClassifier

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

# ── Load model ─────────────────────────────────────────────
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

num_classes = len(le.classes_)
model = PunchClassifier(input_size=26, num_classes=num_classes)
model.load_state_dict(torch.load('punch_classifier_best.pt'))
model.eval()
print(f"Model loaded — classes: {list(le.classes_)}")

# ── Globals ────────────────────────────────────────────────
latest_result = None
frame_buffer  = []
timestamp_ms  = 0
frame_counter = 0
# Smoothing — majority vote over last N predictions
SMOOTH_WINDOW  = 5
CONF_THRESHOLD = 0.55
pred_history   = deque(maxlen=SMOOTH_WINDOW)

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

cap = cv2.VideoCapture(1)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── MediaPipe ──────────────────────────────────────
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (432, 368))
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

            # ── Feature extraction ─────────────────────────
            features = extract_features(landmarks)
            frame_buffer.append(features)
            if len(frame_buffer) > 30:
                frame_buffer.pop(0)

            # ── Model inference ────────────────────────────
            if len(frame_buffer) == 30:
                X = torch.tensor([frame_buffer], dtype=torch.float32)
                with torch.no_grad():
                    logits = model(X)
                    probs  = torch.softmax(logits, dim=1)[0]
                    conf   = probs.max().item()
                    pred   = probs.argmax().item()
                    label  = le.inverse_transform([pred])[0]

                # Only count confident predictions
                if conf >= CONF_THRESHOLD:
                    pred_history.append(label)
                else:
                    pred_history.append('none')

                # Majority vote over smoothing window
                if pred_history:
                    smoothed = max(set(pred_history), key=pred_history.count)

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

        cv2.putText(frame, "Q = quit", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.imshow('Boxing Coach', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
