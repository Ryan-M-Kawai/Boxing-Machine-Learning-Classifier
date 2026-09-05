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
    
    return window_width, window_height, camera_width, camera_height

def hands_up(landmarks):
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    left_hand = landmarks[22]
    right_hand = landmarks[21]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_elbow = landmarks[13]
    right_elbow = landmarks[14]

    left_mouth = landmarks[9]
    right_mouth = landmarks[10]
    
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    left_hand_raised = left_hand.y < left_shoulder.y
    right_hand_raised = right_hand.y < right_shoulder.y
    hands_up = left_hand_raised and right_hand_raised
    # check if wrists are close to shoulders
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    sideways = shoulder_width < 0.15 

    #check if hands are close to mouth
    left_dist = np.linalg.norm([left_hand.x - left_mouth.x, left_hand.y - left_mouth.y])
    right_dist = np.linalg.norm([right_hand.x - right_mouth.x, right_hand.y - right_mouth.y])
    #check for elbows on chin
    left_elbow_dist = np.linalg.norm([left_elbow.x - left_mouth.x, left_elbow.y - left_mouth.y])
    right_elbow_dist = np.linalg.norm([right_elbow.x - right_mouth.x, right_elbow.y - right_mouth.y])
    chin_protected = (left_dist < shoulder_width *0.4 or right_dist < shoulder_width *0.4) or (left_elbow_dist < shoulder_width *0.4 or right_elbow_dist < shoulder_width *0.4)

    #check if shoulder close to chin
    shoulder_to_chin_left = np.linalg.norm([left_shoulder.x - left_mouth.x, left_shoulder.y - left_mouth.y])
    shoulder_to_chin_right = np.linalg.norm([right_shoulder.x - right_mouth.x, right_shoulder.y - right_mouth.y])

    
    left_body_good = angle( 
        get_landmark_coordinates(left_hip, 1, 1), 
        get_landmark_coordinates(left_shoulder, 1, 1), 
        get_landmark_coordinates(left_elbow, 1, 1)) < 60

    right_body_good = angle( 
        get_landmark_coordinates(right_hip, 1, 1), 
        get_landmark_coordinates(right_shoulder, 1, 1), 
        get_landmark_coordinates(right_elbow, 1, 1)) < 60

    body_protected = left_body_good and right_body_good
    


    good_stance = (left_hand_raised and right_hand_raised) and chin_protected
    if sideways:
        if hands_up:
            return "good gaurd"
        if hands_up and ~(left_body_good and right_body_good):
            return "cover your body too"
        else:
            return "cover your chin"
    else:
        if good_stance and body_protected:
            return "good gaurd"
        elif good_stance and not body_protected:
            return "cover your body too"
       
        else:
            return "cover your chin"