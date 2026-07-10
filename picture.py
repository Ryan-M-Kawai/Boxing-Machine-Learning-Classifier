#always start with source venv/Scripts/activate
import cv2 #4.13.0
import mediapipe as mp #0.10.35
import numpy as np #2.2.6
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image
from IPython.display import Video

from helper_functions import get_landmark_coordinates, angle
test_photo = "Pagliacci.PNG"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    #persons direction, mirrored to viewers left and right
    (11,12), #shoulder lines
    (11,13), #left bicep
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


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    for pose_landmarks in pose_landmarks_list:
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmark dots
        for idx, lm in enumerate(pose_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_image, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(annotated_image, str(idx), (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return annotated_image


model_path = 'pose_landmarker_heavy.task'
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

with PoseLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image.create_from_file(test_photo)

    result = landmarker.detect(mp_image)
    pose_landmarker_result = landmarker.detect(mp_image)
    print("Number of people detected:", len(pose_landmarker_result.pose_landmarks))

    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)

    # mp.Image is RGB, cv2 needs BGR
    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    print("point 0 cord", get_landmark_coordinates(result.pose_landmarks[0][0], 640, 480))
    angle("right elbow", get_landmark_coordinates(result.pose_landmarks[0][12], 640, 480),get_landmark_coordinates(result.pose_landmarks[0][14], 640, 480),get_landmark_coordinates(result.pose_landmarks[0][16], 640, 480))
    angle("left elbow", get_landmark_coordinates(result.pose_landmarks[0][11], 640, 480),get_landmark_coordinates(result.pose_landmarks[0][13], 640, 480),get_landmark_coordinates(result.pose_landmarks[0][15], 640, 480))

    cv2.imshow("lmao", annotated_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(pose_landmarker_result)
