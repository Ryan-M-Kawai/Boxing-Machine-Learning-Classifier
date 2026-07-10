import cv2 #4.13.0
import mediapipe as mp #0.10.35
import numpy as np #2.2.6
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