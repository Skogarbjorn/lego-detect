import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_JSON = os.path.join(CURRENT_DIR, "..", "..", "output", "raw.json")
camera_calibration_file = os.path.join(CURRENT_DIR, "..", "..", "misc", "camera_calibration.npz")

data = np.load(camera_calibration_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

def convertToMarkerCoords(point, T_camera_to_marker):
    x,y = point
    x_norm = (x - camera_matrix[0,2]) / camera_matrix[0,0]
    y_norm = (y - camera_matrix[1,2]) / camera_matrix[1,1]

    ray_camera = np.array([x_norm, y_norm, 1.0])

    ray_marker = T_camera_to_marker[:3, :3] @ ray_camera

    t = -T_camera_to_marker[2,3] / ray_marker[2]
    p_marker = T_camera_to_marker[:3,3] + t * ray_marker

    return p_marker

