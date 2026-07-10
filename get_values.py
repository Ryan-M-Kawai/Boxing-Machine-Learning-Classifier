import cv2
import numpy as np
from helper_functions import angle, lm_to_list, lm_to_list_2d
import math
# Logitech C310 HD Webcam fixed focal length = 2.33mm
# height of person in real world = 172.7 cm
# height of camera is 87.6 mm
# ── Camera & body calibration
FOCAL_LENGTH_MM       = 4.4      # from webcam spec sheet
SENSOR_WIDTH_MM       = 4.8      # CHECK this against your webcam's actual datasheet
REAL_WORLD_HEIGHT_CM  = 172.7
CAMERA_HEIGHT_CM      = 87.6
CAMERA_TILT_DEG       = 0.0      # measure/estimate the downward tilt of your webcam

# Torso rectangle — anthropometric fractions of height (swap in real
# tape-measure numbers here if you have them, for better accuracy)
SHOULDER_HALF_WIDTH_CM = REAL_WORLD_HEIGHT_CM * 0.129
HIP_HALF_WIDTH_CM      = REAL_WORLD_HEIGHT_CM * 0.091
TORSO_LENGTH_CM        = REAL_WORLD_HEIGHT_CM * 0.288

MODEL_POINTS = np.array([
    [-SHOULDER_HALF_WIDTH_CM, 0, 0],                 # left shoulder
    [ SHOULDER_HALF_WIDTH_CM, 0, 0],                 # right shoulder
    [-HIP_HALF_WIDTH_CM, -TORSO_LENGTH_CM, 0],       # left hip
    [ HIP_HALF_WIDTH_CM, -TORSO_LENGTH_CM, 0],       # right hip
], dtype=np.float32)

UPPER_ARM_CM = REAL_WORLD_HEIGHT_CM * 0.186   # shoulder -> elbow
FOREARM_CM   = REAL_WORLD_HEIGHT_CM * 0.146   # elbow -> wrist


def get_camera_matrix(image_width_px, image_height_px):
    f = FOCAL_LENGTH_MM * image_width_px / SENSOR_WIDTH_MM   # px
    cx, cy = image_width_px / 2, image_height_px / 2
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)


def get_torso_pose(landmarks, image_width_px, image_height_px, camera_matrix):
    """Returns (torso_center_camera_cm, R) via PnP, or (None, None) if it fails."""
    image_points = np.array([
        [landmarks[11].x * image_width_px, landmarks[11].y * image_height_px],
        [landmarks[12].x * image_width_px, landmarks[12].y * image_height_px],
        [landmarks[23].x * image_width_px, landmarks[23].y * image_height_px],
        [landmarks[24].x * image_width_px, landmarks[24].y * image_height_px],
    ], dtype=np.float32)

    dist_coeffs = np.zeros(4, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return tvec.flatten(), R


def pixel_to_ray(u, v, camera_matrix):
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return ray / np.linalg.norm(ray)


def solve_child_depth(parent_3d, ray_dir_unit, bone_length_cm, prev_child_3d=None):
    """Intersects a camera ray with a sphere of radius bone_length_cm around parent_3d."""
    a = np.dot(ray_dir_unit, ray_dir_unit)
    b = -2 * np.dot(ray_dir_unit, parent_3d)
    c = np.dot(parent_3d, parent_3d) - bone_length_cm ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None  # 2D observation inconsistent with bone length -> noisy frame
    sq = np.sqrt(disc)
    t1, t2 = (-b + sq) / (2 * a), (-b - sq) / (2 * a)
    cand1, cand2 = t1 * ray_dir_unit, t2 * ray_dir_unit
    if prev_child_3d is not None:
        d1 = np.linalg.norm(cand1 - prev_child_3d)
        d2 = np.linalg.norm(cand2 - prev_child_3d)
        return cand1 if d1 < d2 else cand2
    return cand1 if cand1[2] > cand2[2] else cand2  # fallback: farther point


def get_wrist_depth(landmarks, side, image_width_px, image_height_px,
                     camera_matrix, torso_center, R, prev_elbow=None, prev_wrist=None):
    """side: 'L' or 'R'. Returns (elbow_3d, wrist_3d) in camera-frame cm, or (None, None)."""
    shoulder_idx, elbow_idx, wrist_idx = {'L': (11, 13, 15), 'R': (12, 14, 16)}[side]
    shoulder_local = MODEL_POINTS[0] if side == 'L' else MODEL_POINTS[1]
    shoulder_3d = torso_center + R @ shoulder_local

    eu = landmarks[elbow_idx].x * image_width_px
    ev = landmarks[elbow_idx].y * image_height_px
    elbow_ray = pixel_to_ray(eu, ev, camera_matrix)
    elbow_3d = solve_child_depth(shoulder_3d, elbow_ray, UPPER_ARM_CM, prev_elbow)
    if elbow_3d is None:
        return None, None

    wu = landmarks[wrist_idx].x * image_width_px
    wv = landmarks[wrist_idx].y * image_height_px
    wrist_ray = pixel_to_ray(wu, wv, camera_matrix)
    wrist_3d = solve_child_depth(elbow_3d, wrist_ray, FOREARM_CM, prev_wrist)

    return elbow_3d, wrist_3d
#users left and right
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
    
    sideways = shoulder_x_diff <0.15 and hip_x_diff <0.15
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
    return stance
    return shoulder_x_diff
    return hip_x_diff
    return sideways

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

    hip_x_diff = hipL.x - hipR.x

    return [
        foot_x_diff,
        foot_z_diff,
        shoulder_x_diff,
        shoulder_z_diff,
        hip_x_diff
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
    wristLextension_forward = wristL.z - shoulderL.z
    wristRextension_forward = wristR.z - shoulderR.z
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

