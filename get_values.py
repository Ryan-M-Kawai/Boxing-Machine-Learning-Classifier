import cv2
import numpy as np
from helper_functions import angle, lm_to_list, lm_to_list_2d
from get_z import FOCAL_LENGTH_PX, shoulder_to_wrist_R, shoulder_to_wrist_L, get_shoulder_z
import math
# Logitech C310 HD Webcam fixed focal length = 2.33mm
# height of person in real world = 172.7 cm
# height of camera is 87.6 mm
# ── Camera & body calibration
#img height and width 
image_width_px = 640
image_height_px = 360
#users left and right
# module-level, outside any function — persists across frames


def get_stance(landmarks):
    shoulderL = landmarks[11]
    shoulderR = landmarks[12]
    footL = landmarks[31]
    footR = landmarks[32]
    hipL = landmarks[23]
    hipR = landmarks[24]
    #when sideways facing right, orthodox is positive southpaw negative
    #when sideways facing left, orthodox is negative southpaw is positive
    #irrelevant when facing forward
    foot_x_diff = footL.x - footR.x
    if(foot_x_diff <=0): #
        foot_x_diff_bool = 0
    else:
        foot_x_diff_bool = 1

    # which foot is forward, pos if left foot back, neg if left foot forward
    # when facing camera: if left foot forward negative, orthodox is negative, southpaw is positive
    # Camera view sideways: if facing right, positive, if facing left: negative
    foot_z_diff = footL.z - footR.z
    if(foot_z_diff <=0):
        foot_z_diff_bool = 0
    else:
        foot_z_diff_bool =1
    #same logic as foot
    hip_z_diff = hipL.z - hipR.z
    if(hip_z_diff <= 0):
        hip_z_diff_bool = 0
    else:
        hip_z_diff_bool =1

    #positive sideways facing right, negative sideways facing left
    #when forwards, positive is orthodox, negative is southpaw
    shoulder_z_diff = shoulderL.z - shoulderR.z
    if shoulder_z_diff <= 0:
        shoulder_z_diff_bool = 0 #right
    else:
        shoulder_z_diff_bool = 1 #left
    
    shoulder_x_diff = abs(shoulderL.x - shoulderR.x) 
    hip_x_diff = abs(hipL.x-hipR.x)
    
    sideways = shoulder_x_diff <0.11 and hip_x_diff <0.11
    #print(f"foot_z_diff: {foot_z_diff:.3f}  shoulder_z_diff: {shoulder_z_diff:.3f}  sideways: {sideways}")
    if sideways == 1:
        #print("side")
        #0 if right, 1 if left
        direction = shoulder_z_diff_bool
        stance = direction ^ foot_x_diff_bool
    else:
        #print("forward")
        stance_float = (hip_z_diff + shoulder_z_diff)/2
        if stance_float <=0:
            stance = 0
        else: 
            stance = 1
    #talk abt k maps
    #0 = orthodox, 1 = southpaw
    #return stance
    #return shoulder_x_diff
    #return hip_x_diff
    return shoulder_z_diff_bool
def direction_facing(landmarks):
    #just return if sideways or not, 0 = right, 1 = left
    # shoulderL = landmarks[11]
    # shoulderR = landmarks[12]
    # footL = landmarks[31]
    # footR = landmarks[32]
    # hipL = landmarks[23]
    # hipR = landmarks[24]
    # wristL = landmarks[15]
    # wristR = landmarks[16]
    # direction = None
    # shoulder_x_diff = abs(shoulderL.x - shoulderR.x) 
    # hip_x_diff = abs(hipL.x-hipR.x)
    # leg_x_diff = footL.x - footR.x
    # shoulder_z_diff = shoulderL.z- shoulderR.z
    # shoulder_mid_y = (shoulderL.y + shoulderR.y) / 2
    # hip_mid_y = (hipL.y + hipR.y) / 2
    # torso_height = abs(hip_mid_y - shoulder_mid_y) + 1e-6  # avoid div by zero

    # shoulder_ratio = shoulder_x_diff / torso_height
    # hip_ratio = hip_x_diff / torso_height

    # sideways = shoulder_ratio < 0.5 and hip_ratio < 0.5

    # hands_right = wristR.x > shoulderR.x and wristL.x > shoulderL.x
    # hands_left = wristL.x < shoulderL.x and wristR.x < shoulderR.x
    # if sideways:
    #     if shoulder_z_diff <= 0 and leg_x_diff >=0 and hands_right:
    #         direction = 0 #right
    #     elif shoulder_z_diff > 0 and leg_x_diff <=0 and hands_left:
    #         direction = 1 #left
    # else:
    #     direction = 2 #forwards

    # 0 = right, 1 = left, 2 = forward
    shoulderL = landmarks[11]
    shoulderR = landmarks[12]
    hipL = landmarks[23]
    hipR = landmarks[24]

    shoulder_x_diff = abs(shoulderL.x - shoulderR.x)
    hip_x_diff = abs(hipL.x - hipR.x)
    shoulder_z_diff = shoulderL.z - shoulderR.z

    shoulder_mid_y = (shoulderL.y + shoulderR.y) / 2
    hip_mid_y = (hipL.y + hipR.y) / 2
    torso_height = abs(hip_mid_y - shoulder_mid_y) + 1e-6

    shoulder_ratio = shoulder_x_diff / torso_height
    hip_ratio = hip_x_diff / torso_height

    sideways = shoulder_ratio < 0.3 and hip_ratio < 0.3
    if sideways:
        direction = 0 if shoulder_z_diff <= 0 else 1
    else:
        direction = 2
    return direction

def get_stance_features(landmarks):
    shoulderL = landmarks[11]
    shoulderR = landmarks[12]
    footL = landmarks[31]
    footR = landmarks[32]
    hipL = landmarks[23]
    hipR = landmarks[24]
    
    foot_x_diff = footL.x - footR.x
    foot_z_diff = footL.z - footR.z

    shoulder_x_diff = shoulderL.x - shoulderR.x
    shoulder_z_diff = shoulderL.z - shoulderR.z
    wristLextension_forward = shoulder_to_wrist_L(landmarks)
    wristRextension_forward = shoulder_to_wrist_R(landmarks)
    hip_x_diff = hipL.x - hipR.x
    direction = direction_facing(landmarks)
    return [
        foot_x_diff,
        foot_z_diff,
        shoulder_x_diff,
        shoulder_z_diff,
        wristLextension_forward,
        wristRextension_forward,
        hip_x_diff,
        direction
    ]



def extract_features(landmarks):


    chinL = landmarks[9]
    chinR = landmarks[10] 

    shoulderL = landmarks[11]
    shoulderR = landmarks[12]

    elbowL = landmarks[13]
    elbowR = landmarks[14]

    wristL = landmarks[15]
    wristR = landmarks[16]

    hipL =landmarks[23]
    hipR = landmarks[24]

    kneeL = landmarks[25]
    kneeR = landmarks[26]

    shinL = landmarks[27]
    shinR = landmarks[28]

    shoulder_width = np.linalg.norm([shoulderL.x - shoulderR.x, shoulderL.y - shoulderR.y])
    body_height = abs(((chinR.y+chinL.y)/2) - ((hipL.y + hipR.y) / 2)) + 1e-6

    #angles
    elbowAngleL   = angle(lm_to_list_2d(shoulderL), lm_to_list_2d(elbowL),   lm_to_list_2d(wristL))
    elbowAngleR   = angle(lm_to_list_2d(shoulderR), lm_to_list_2d(elbowR),   lm_to_list_2d(wristR))
    shoulderAngleL = angle(lm_to_list_2d(elbowL),  lm_to_list_2d(shoulderL), lm_to_list_2d(hipL))
    shoulderAngleR = angle(lm_to_list_2d(elbowR),  lm_to_list_2d(shoulderR), lm_to_list_2d(hipR))
    legAngleL     = angle(lm_to_list(hipL),     lm_to_list(kneeL),     lm_to_list(shinL))
    legAngleR     = angle(lm_to_list(hipR),     lm_to_list(kneeR),     lm_to_list(shinR))
    hipAngleR     = angle(lm_to_list(shoulderL),lm_to_list(hipL),      lm_to_list(kneeL))
    hipAngleL     = angle(lm_to_list(shoulderR),lm_to_list(hipR),      lm_to_list(kneeR))
    #height of wrists compared to shoulders
    wristHeightL = (shoulderL.y - wristL.y)/body_height
    wristHeightR = (shoulderR.y - wristR.y)/body_height

    #wrist and elbow position relative to shoulder center
    shoulderMidX = (shoulderL.x + shoulderR.x) / 2
    wristLextension_horz = (wristL.x - shoulderMidX) / (shoulder_width + 1e-6)
    wristRextension_horz = (wristR.x - shoulderMidX) / (shoulder_width + 1e-6)

    elbowHeightL = (shoulderL.y - elbowL.y) / body_height
    elbowHeightR = (shoulderR.y - elbowR.y) / body_height   

    #forward punching
    wristLextension_forward = shoulder_to_wrist_L(landmarks)
    wristRextension_forward = shoulder_to_wrist_R(landmarks)
    wrist_foot_extensionL = wristL.z - shinL.z
    wrist_foot_extensionR = wristR.z - shinR.z

    wristAboveElbowL = (elbowL.y - wristL.y) / body_height  # positive = wrist higher than elbow
    wristAboveElbowR = (elbowR.y - wristR.y) / body_height

    #shoulder tilt
    shoulderTilt = (shoulderL.y - shoulderR.y)/shoulder_width + 1e-6
    
    #feet depth z
    
    #hip rotation for hooks
    hip_width = np.linalg.norm([hipL.x - hipR.x, hipL.y - hipR.y]) + 1e-6
    hip_tilt  = (hipL.y - hipR.y) / hip_width

    #wrist and elbow x compared to shoulder
    wristLateralL = (wristL.x - shoulderL.x) / (shoulder_width + 1e-6)
    wristLateralR = (wristR.x - shoulderR.x) / (shoulder_width + 1e-6)
    elbowFlareL = (elbowL.x - shoulderL.x) / (shoulder_width + 1e-6)
    elbowFlareR = (elbowR.x - shoulderR.x) / (shoulder_width + 1e-6)

    #stance
    stance = get_stance(landmarks)
    #16 outputs
    return [
    elbowAngleL,            # 0  - left elbow bend
    elbowAngleR,            # 1  - right elbow bend
    shoulderAngleL,         # 2  - left shoulder raise
    shoulderAngleR,         # 3  - right shoulder raise
    wristHeightL,           # 4  - left hand height
    wristHeightR,           # 5  - right hand height
    wristLextension_horz,   # 6  - left wrist horizontal
    wristRextension_horz,   # 7  - right wrist horizontal
    wristLextension_forward,# 8  - left wrist depth
    wristRextension_forward,# 9  - right wrist depth 
    shoulderTilt,           # 10 - shoulder level
    hip_tilt,               # 11 - hip rotation
    hipAngleL,              # 12 - left hip angle
    hipAngleR,              # 13 - right hip angle
    legAngleL,              # 14 - left leg angle
    legAngleR,              # 15 - right leg angle
    wristAboveElbowL,       # 16 - left wrist height from elbow
    wristAboveElbowR,       # 17 - right wrist height from elbow
    elbowHeightL,           # 18 - left elbow height compared to shoulder 
    elbowHeightR,           # 19 - right elbow height compared to shoulder 
    wristLateralL,          # 20 - left wrist x from shoulder
    wristLateralR,          # 21 - right wrist x from shoulder
    elbowFlareL,            # 22 - left elbow x from shoulder
    elbowFlareR,            # 23 - right elbow x from shoulder
    wrist_foot_extensionL,  # 24 - wrist z vs shin z L
    wrist_foot_extensionR,  # 25 - wrist z vs shin z R
    ]

