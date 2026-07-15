import cv2
import numpy as np
from helper_functions import angle, lm_to_list, lm_to_list_2d
from collections import defaultdict
import math
# height of person in real world = 172.7 cm
# height of camera is 87.6 cm
# ── Camera & body calibration
import numpy as np

# ── Calibration (measure once) ──────────────────────────────
REAL_WORLD_HEIGHT_CM = 172.7     # person's height, tip of head to floor
CAMERA_HEIGHT_CM     = 87.6      # camera's height off the floor
FOCAL_LENGTH_PX       = 429.2     # derived earlier from FoV at your processing resolution (432x368)

# Anthropometric fractions of height (Drillis & Contini) — used since you
# can't derive true bone length from landmarks alone (that's the
# depth/scale ambiguity problem from before)
UPPER_ARM_CM = 29   # shoulder -> elbow
FOREARM_CM   = 24 # elbow -> wrist


def pixel_to_ray(u, v, image_width_px, image_height_px, focal_length_px):
    """3D unit direction from the camera through pixel (u, v)."""
    cx, cy = image_width_px / 2, image_height_px / 2
    ray = np.array([(u - cx) / focal_length_px, (v - cy) / focal_length_px, 1.0])
    return ray / np.linalg.norm(ray)


def get_person_distance_cm(landmarks,x, image_height_px, focal_length_px):
    """
    Similar-triangles distance estimate using apparent full-body height.
    Non-noisy because total height barely changes frame-to-frame — unlike
    wrist position, which is exactly what we're trying to measure.
    """
    # nose (0) to ankle midpoint (27/28) as a full-body proxy
    top_y = landmarks[0].y * image_height_px
    bottom_y = (landmarks[27].y + landmarks[28].y) / 2 * image_height_px
    apparent_height_px = abs(bottom_y - top_y)

    if apparent_height_px < 1e-6:
        return None
    return (REAL_WORLD_HEIGHT_CM * focal_length_px) / apparent_height_px


def solve_child_depth(parent_3d, ray_dir_unit, bone_length_cm, prev_child_3d=None):
    """Intersect a camera ray with a sphere of known radius around the parent joint."""
    a = np.dot(ray_dir_unit, ray_dir_unit)
    b = -2 * np.dot(ray_dir_unit, parent_3d)
    c = np.dot(parent_3d, parent_3d) - bone_length_cm ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None  # bone-length constraint inconsistent with this frame — skip it
    sq = np.sqrt(disc)
    t1, t2 = (-b + sq) / (2 * a), (-b - sq) / (2 * a)
    cand1, cand2 = t1 * ray_dir_unit, t2 * ray_dir_unit
    if prev_child_3d is not None:
        return cand1 if np.linalg.norm(cand1 - prev_child_3d) < np.linalg.norm(cand2 - prev_child_3d) else cand2
    return cand1 if cand1[2] > cand2[2] else cand2


def get_wrist_z(landmarks, side, image_width_px, image_height_px,
                 focal_length_px, prev_elbow=None, prev_wrist=None):
    """
    side: 'L' or 'R'. Returns wrist z (distance from camera, cm), or None.
    """
    shoulder_idx, elbow_idx, wrist_idx = {'L': (11, 13, 15), 'R': (12, 14, 16)}[side]

    person_z = get_person_distance_cm(landmarks, image_width_px, image_height_px, focal_length_px)
    if person_z is None:
        return None

    # anchor shoulder in 3D: same distance as the person overall, along the
    # ray through the shoulder's own pixel position
    su = landmarks[shoulder_idx].x * image_width_px
    sv = landmarks[shoulder_idx].y * image_height_px
    shoulder_ray = pixel_to_ray(su, sv, image_width_px, image_height_px, focal_length_px)
    shoulder_3d = shoulder_ray * (person_z / shoulder_ray[2])

    # walk to elbow
    eu = landmarks[elbow_idx].x * image_width_px
    ev = landmarks[elbow_idx].y * image_height_px
    elbow_ray = pixel_to_ray(eu, ev, image_width_px, image_height_px, focal_length_px)
    elbow_3d = solve_child_depth(shoulder_3d, elbow_ray, UPPER_ARM_CM, prev_elbow)
    if elbow_3d is None:
        return None

    # walk to wrist
    wu = landmarks[wrist_idx].x * image_width_px
    wv = landmarks[wrist_idx].y * image_height_px
    wrist_ray = pixel_to_ray(wu, wv, image_width_px, image_height_px, focal_length_px)
    wrist_3d = solve_child_depth(elbow_3d, wrist_ray, FOREARM_CM, prev_wrist)
    if wrist_3d is None:
        return None

    return wrist_3d[2], elbow_3d, wrist_3d  # z = distance from camera, cm