import cv2 #4.13.0
import mediapipe as mp #0.10.35
import numpy as np #2.2.6
import pyautogui #0.9.53
import math #3.11.4
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def get_landmark_coordinates(landmark, w , h):
    return int(landmark.x * w), int(landmark.y * h), landmark.z

def angle (a,b,c):
    # Calculate the angle between three points a, b, and c
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_rad = np.arccos(cosine_angle)
    
    return np.degrees(angle_rad)

def draw_debug(frame, msg, row, color=(0,255,255)):
    cv2.putText(frame, msg, (10, 30 + row*20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
def lm_to_list(lm):
    return [lm.x, lm.y, lm.z]

def lm_to_list_2d(lm):
    return [lm.x, lm.y]  # ignore z

def resize_window_to_screen(cap):
    # Get camera resolution
    camera_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    camera_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = camera_width / camera_height
    print(f"Camera Resolution: {int(camera_width)}x{int(camera_height)}")
    print(f"Camera Aspect Ratio: {aspect_ratio:.2f}")

    # Get screen resolution
    cpu_width, cpu_height = pyautogui.size()
    print(f"Screen Resolution: {cpu_width}x{cpu_height}")

    # Leave a margin so the window doesn't butt right up against screen edges
    # (taskbar, title bar, etc.) — adjust to taste
    MARGIN = 0.9
    max_width  = cpu_width * MARGIN
    max_height = cpu_height * MARGIN

    # Scale factor: whichever dimension is the tighter constraint wins
    scale = min(max_width / camera_width, max_height / camera_height)

    window_width  = math.floor(camera_width * scale)
    window_height = math.floor(camera_height * scale)

    print(f"Window Width: {window_width}, Window Height: {window_height}")
    
    return window_width, window_height
