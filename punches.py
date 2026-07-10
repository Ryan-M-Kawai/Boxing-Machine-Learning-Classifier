import cv2 #4.13.0
import mediapipe as mp #0.10.35
import numpy as np #2.2.6
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
#from IPython.display import Video

from helper_functions import get_landmark_coordinates, angle, draw_debug

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
last_punch = None
stance = None
def stance(landmarks):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    left_hand = landmarks[22]
    right_hand = landmarks[21]
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)

    sideways = shoulder_width < 0.15 
    if sideways:
        if left_shoulder.z < right_shoulder.z:
            if left_shoulder.x < right_shoulder.x:
                return "orthodox stance"
            else:
                return "southpaw stance"
        
        elif  left_shoulder.z > right_shoulder.z:
            if right_shoulder.x < left_shoulder.x:
                return "orthodox stance r"
            else:
                return "southpaw stance r"
    else:
        z_diff = left_shoulder.z - right_shoulder.z
        elbow_diff = landmarks[13].z - landmarks[14].z  # left_elbow - right_elbow

        if z_diff > 0.05 or elbow_diff > 0.05:
            return "southpaw stance"
        elif z_diff < -0.05 or elbow_diff < -0.05:
            return "orthodox stance"
        else:
            return "none"
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
    


    good_philly = (shoulder_to_chin_left < shoulder_width * 0.5 and right_hand_raised) or (shoulder_to_chin_right < shoulder_width * 0.5 and left_hand_raised)
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
        elif good_philly:
            return "good Philly shell"
        else:
            return "cover your chin"
def punch(landmarks):
    left_knee = landmarks[25]
    right_knee = landmarks[26]

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
    if sideways:
        if left_shoulder.z < right_shoulder.z:
            #orthodox
            if left_shoulder.x < right_shoulder.x:
                if angle("left hand",
                         get_landmark_coordinates(left_hand,1,1),
                         get_landmark_coordinates(left_elbow,1,1),
                         get_landmark_coordinates(left_shoulder,1,1)) > 60:
                    return "jab"
                if angle("right hand",
                         get_landmark_coordinates(right_hand,1,1),
                         get_landmark_coordinates(right_elbow,1,1),
                         get_landmark_coordinates(right_shoulder,1,1)) > 60:
                    return "cross"
            else:
                return None
        
        elif  left_shoulder.z > right_shoulder.z:
            if right_shoulder.x < left_shoulder.x:
                if angle("right hand",
                         get_landmark_coordinates(right_hand,1,1),
                         get_landmark_coordinates(right_elbow,1,1),
                         get_landmark_coordinates(right_shoulder,1,1)) > 60:
                    return "cross"
            else:
                if angle("left hand",
                         get_landmark_coordinates(left_hand,1,1),
                         get_landmark_coordinates(left_elbow,1,1),
                         get_landmark_coordinates(left_shoulder,1,1)) > 60:
                    return "jab"
                
    else:
        if (left_shoulder.z > right_shoulder.z)or (left_knee.z > right_knee.z):
            if angle("left hand",
                         get_landmark_coordinates(left_hand,1,1),
                         get_landmark_coordinates(left_elbow,1,1),
                         get_landmark_coordinates(left_shoulder,1,1)) > 60:
                    return "cross"
            if angle("right hand",
                         get_landmark_coordinates(right_hand,1,1),
                         get_landmark_coordinates(right_elbow,1,1),
                         get_landmark_coordinates(right_shoulder,1,1)) > 60:
                    return "jab"
        else:
            if angle("left hand",
                         get_landmark_coordinates(left_hand,1,1),
                         get_landmark_coordinates(left_elbow,1,1),
                         get_landmark_coordinates(left_shoulder,1,1)) > 60:
                    return "jab"
            if angle("right hand",
                         get_landmark_coordinates(right_hand,1,1),
                         get_landmark_coordinates(right_elbow,1,1),
                         get_landmark_coordinates(right_shoulder,1,1)) > 60:
                    return "cross"