import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import cv2

# Open the default camera (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    camera_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    camera_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = camera_width / camera_height
    print(f"Resolution: {int(camera_width)}x{int(camera_height)}")
    print(f"Aspect Ratio: {aspect_ratio:.2f}")

cap.release()

cpu_width, cpu_height = pyautogui.size()
print(f"Screen: {cpu_width}x{cpu_height}")

MARGIN = 0.9
max_width  = cpu_width * MARGIN
max_height = cpu_height * MARGIN

# Scale factor: whichever dimension is the tighter constraint wins
scale = min(max_width / camera_width, max_height / camera_height)

window_width  = math.floor(camera_width * scale)
window_height = math.floor(camera_height * scale)

print(f"Window Width: {window_width}, Window Height: {window_height}")
print("pyautogui:", pyautogui.__version__)
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__)
