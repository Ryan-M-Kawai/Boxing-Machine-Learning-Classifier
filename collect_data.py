#always start with source venv/Scripts/activate
import cv2 #4.13.0
import mediapipe as mp #0.10.35
import numpy as np #2.2.6
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from helper_functions import get_landmark_coordinates, angle, draw_debug 
from punches import hands_up, stance, punch
from get_z import shoulder_to_wrist_L, shoulder_to_wrist_R
from get_values import direction_facing, extract_features, get_stance, get_stance_features
from collections import deque
import json
import pylance
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    #persons direction, mirrored to viewers left and right
    (11,13), #left bicep
    (11,12), #shoulders
    (13,15), #left forearm
    (12,14), #right bicep
    (14,16), #right forearm
    (11,23), #left torso line
    (12,24), # right torso line
    (23,24), # pelvis
    (23,25), # left thigh IMPORTANT 25 is left knee
    (25,27), # left shin
    (24,26), # right thigh IMPORTANT 26 is right knee
    (26,28), #right shin
    (27,29), #left heel??
    (29,31), #left foot
    (28,30), #right heel
    (30,32), #right foott
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8), 
    (9,10),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22) # hands/mouth
]
model_path = 'pose_landmarker_full.task'
static_image_mode = False
max_num_hands = 2
max_num_people = 1

latest_timestamp = 0
frames_detected = 0
frames_sent = 0

#should make tracking better?
latest_result = None
min_detection_confidence = 0.3
min_tracking_confidence = 0.6
timestamp_ms = int(time.time() * 1000)
#img with and height
img_width = 640
img_height = 360
#timer
print_timer_start = None  # add to globals at top

#training
frame_buffer = []

#for stances
stance_history = deque(maxlen=10)
stance_recording = False
current_stance_label = None
last_stance_save = 0
STANCE_SAVE_INTERVAL = 2.0  # seconds

# ── NEW: orientation tracking for frontal/sideways split ────
direction_history = deque(maxlen=10)   # smooths direction_facing() over recent frames
current_direction = 2                  # 0=right facing, 1=left facing, 2/other=forwards
current_sideways  = False              # True when current_direction is 0 or 1

try:
    with open('stance_data.json') as f:
        stance_data = json.load(f)
    print(f"[LOADED] {len(stance_data)} existing stance samples from stance_data.json")
except FileNotFoundError:
    stance_data = []
    print("[NEW] No existing stance data found, starting fresh")
try:
    with open('training_data.json') as f:
        training_data = json.load(f)
    print(f"[LOADED] {len(training_data)} existing reps from training_data.json")
except FileNotFoundError:
    training_data = []
    print("[NEW] No existing data found, starting fresh")

# ── NEW: separate sideways dataset ───────────────────────────
try:
    with open('sideways_training_data.json') as f:
        sideways_training_data = json.load(f)
    print(f"[LOADED] {len(sideways_training_data)} existing reps from sideways_training_data.json")
except FileNotFoundError:
    sideways_training_data = []
    print("[NEW] No existing sideways data found, starting fresh")

current_label = None
recording = False

#p and s already taken
LABELS = {
    #from watcher left and right
    ord('q'): 'jab',
    ord('w'): 'cross', 
    ord('e'): 'right_hook',
    ord('r'): 'left_hook', 
    ord('t'): 'right_uppercut',
    ord('y'): 'left_uppercut',
}
STANCES = {
    ord('0'): 'orthodox',
    ord('1'): 'southpaw'
} 

# ── NEW: reusable compact writer for either training set ─────
class CompactEncoder(json.JSONEncoder):
    def iterencode(self, obj, _one_shot=False):
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            yield '[\n'
            for i, rep in enumerate(obj):
                yield '  {\n'
                yield f'    "label": "{rep["label"]}",\n'
                yield '    "frames": [\n'
                for j, frame in enumerate(rep['frames']):
                    comma = ',' if j < len(rep['frames']) - 1 else ''
                    yield f'      {json.dumps([round(v, 3) for v in frame])}{comma}\n'
                yield '    ]\n'
                yield '  }' + (',' if i < len(obj) - 1 else '') + '\n'
            yield ']\n'
        else:
            yield from super().iterencode(obj, _one_shot)

def save_dataset(data, path):
    with open(path, 'w') as f:
        f.write(CompactEncoder().encode(data))

def print_class_breakdown(data, path_label):
    from collections import Counter
    counts = Counter(d['label'] for d in data)
    print(f"  --- all reps so far ({path_label}) ---")
    for label, count in sorted(counts.items()):
        print(f"    {label}: {count} reps")
    print()

def print_result(result:PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result    

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback = print_result)

cap = cv2.VideoCapture(0) #try 1 or 2
timestamp_ms = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break
        
        frames_sent +=1
        #convert BGR to RGB and create mediapipe image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #wrap in Mediapipe Image and send to landmarker    
        #aspect ratio of camera is 1280 x 720
        small_frame = cv2.resize(rgb_frame, (img_width, img_height))  
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = small_frame)

        timestamp_ms +=1
        landmarker.detect_async(mp_image, timestamp_ms)

        h,w,_ = frame.shape
    

        if latest_result and latest_result.pose_landmarks:
            landmarks = latest_result.pose_landmarks[0]  # first person
            # Draw joints
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Draw skeleton
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    x1 = int(landmarks[start_idx].x * w)
                    y1 = int(landmarks[start_idx].y * h)
                    x2 = int(landmarks[end_idx].x * w)
                    y2 = int(landmarks[end_idx].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            stance_history.append(get_stance(landmarks))
            smoothed_stance = max(set(stance_history), key=stance_history.count)

            # shoulder_x_diff = abs(shoulderL.x - shoulderR.x)
            # hip_x_diff = abs(hipL.x - hipR.x)
            # sideways = shoulder_x_diff < 0.11 and hip_x_diff < 0.11

            direction = direction_facing(landmarks)
            if direction == 0:
                draw_debug(frame, f"right facing", 1, (0, 255, 0))
            elif direction == 1:
                draw_debug(frame, f"left facing", 1, (0, 255, 0))
            else:
                draw_debug(frame, f"forwards", 1, (0, 0, 255))

            # ── NEW: smooth direction over recent frames, then derive
            # a single sideways flag used to route saved reps ──
            direction_history.append(direction)
            current_direction = max(set(direction_history), key=direction_history.count)
            current_sideways = current_direction in (0, 1)

            orient_msg = "SIDEWAYS" if current_sideways else "FRONTAL"
            orient_color = (255, 0, 255) if current_sideways else (0, 255, 0)
            draw_debug(frame, orient_msg, 2, orient_color)

            guard_msg = hands_up(landmarks)
            guard_color = (0,255,0) if guard_msg == "good gaurd" else (0,0,255)
            draw_debug(frame, guard_msg, 3, guard_color)

           
            wrist_z_R = shoulder_to_wrist_R(landmarks)
            if wrist_z_R is not None:
                draw_debug(frame, f"wrist_z: {wrist_z_R:.2f} cm", 4, (255, 255, 0))
            else:
                draw_debug(frame, "wrist_z: no reading", 4, (0, 0, 255))
            wrist_z_L = shoulder_to_wrist_L(landmarks)
            if wrist_z_L is not None:
                draw_debug(frame, f"wrist_z: {wrist_z_L:.2f} cm", 5, (255, 255, 0))
            else:
                draw_debug(frame, "wrist_z: no reading", 5, (0, 0, 255))
                
            features = extract_features(landmarks)
            stance_features = get_stance_features(landmarks)

            frame_buffer.append(features)
            if len(frame_buffer) > 30:
                frame_buffer.pop(0)

        if recording:
            # NEW — label shows which dataset the rep is currently headed to
            target_name = "sideways_training_data" if current_sideways else "training_data"
            label_text = f"RECORDING [{target_name}]: {current_label}  |  buffer: {len(frame_buffer)}/30"
            cv2.putText(frame, label_text, (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Frontal: {len(training_data)}  |  Sideways: {len(sideways_training_data)}  |  press label key then S to save",
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(frame, "q=jab w=cross e=r.hook r=l.hook t=r.upper y=l.upper 1=southpaw 0=orthodox p = print features  S=save x=quit",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow('Collect Data', frame)

        # ── Keypress handling ──────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if print_timer_start is not None:
            elapsed = time.time() - print_timer_start
            if int(elapsed) == 1 and int(elapsed) < 2:
                print("2...")
            elif int(elapsed) == 2 and int(elapsed) < 3:
                print("1...")
            elif elapsed >= 3:
                print("\n--- features ---")
                for i, val in enumerate(features):
                    print(f"  [{i}] {val:.4f}")
                print_timer_start = None
        if key in LABELS:
            current_label = LABELS[key]
            recording = True
            print(f"[LABEL] Recording: {current_label}")
    #    #new ---------------------------------------------------
        elif key in STANCES:
            if latest_result and latest_result.pose_landmarks:
                landmarks = latest_result.pose_landmarks[0]
                label = STANCES[key]
                features = get_stance_features(landmarks)

                stance_data.append({
                    'label': label,
                    'features': [round(v, 5) for v in features]
                })

                with open('stance_data.json', 'w') as f:
                    json.dump(stance_data, f, indent=2)

                print(f"[STANCE SAVED] #{len(stance_data)} — {label}: "
                    f"{[round(v, 3) for v in features]}")

                from collections import Counter
                counts = Counter(d['label'] for d in stance_data)
                for lbl, count in sorted(counts.items()):
                    print(f"    {lbl}: {count} samples")
            else:
                print("[WARN] No pose detected — can't snapshot stance")
    #     #-----------------------------------------------------
        elif key == ord('s'):
            if not recording:
                print("[WARN] Press a label key first (q/w/e/r/t/y/1/2) to start recording")
            elif len(frame_buffer) < 30:
                print(f"[WARN] Buffer only {len(frame_buffer)}/30 frames — hold the pose longer")
            #elif stance_recording == True:
                
            else:
                # ── NEW: route the rep to the correct dataset based on
                # current_sideways at the moment of save ──
                if current_sideways:
                    target_data = sideways_training_data
                    target_path = 'sideways_training_data.json'
                else:
                    target_data = training_data
                    target_path = 'training_data.json'

                target_data.append({
                    'frames': list(frame_buffer),
                    'label': current_label
                })
                print(f"[SAVED to {target_path}] Rep #{len(target_data)} — {current_label}")
            
                saved_frames = target_data[-1]['frames']
                print(f"  label     : {current_label}")
                print(f"  frames    : {len(saved_frames)}")
                print(f"  features  : {len(saved_frames[0])}")
                
                print_class_breakdown(target_data, target_path)
                recording = False
                current_label = None

                save_dataset(target_data, target_path)
        elif key == ord('p'):
            print_timer_start = time.time()
            print("snap shot in 3")
        elif key == ord('x'):
            if training_data:
                save_dataset(training_data, 'training_data.json')
                print(f"[DONE] Saved {len(training_data)} frontal reps to training_data.json")
                from collections import Counter
                counts = Counter(d['label'] for d in training_data)
                for label, count in sorted(counts.items()):
                    print(f"  {label}: {count} reps")
            if sideways_training_data:
                save_dataset(sideways_training_data, 'sideways_training_data.json')
                print(f"[DONE] Saved {len(sideways_training_data)} sideways reps to sideways_training_data.json")
                from collections import Counter
                counts = Counter(d['label'] for d in sideways_training_data)
                for label, count in sorted(counts.items()):
                    print(f"  {label}: {count} reps")
            if not training_data and not sideways_training_data:
                print("[DONE] No data collected.")
            break    

        
cap.release()
cv2.destroyAllWindows()