import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_JSON = os.path.join(CURRENT_DIR, "..", "..", "output", "raw.json")
camera_calibration_file = os.path.join(CURRENT_DIR, "..", "..", "misc", "camera_calibration.npz")

data = np.load(camera_calibration_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

def convertToMarkerCoords(point, T_camera_to_marker):
    point = np.asarray(point).ravel()  
    if point.size < 2:
        raise ValueError(f"Point has less than 2 elements: {point}")
    x, y = point[:2]
    x_norm = (x - camera_matrix[0,2]) / camera_matrix[0,0]
    y_norm = (y - camera_matrix[1,2]) / camera_matrix[1,1]

    ray_camera = np.array([x_norm, y_norm, 1.0])

    ray_marker = T_camera_to_marker[:3, :3] @ ray_camera

    t = -T_camera_to_marker[2,3] / ray_marker[2]
    p_marker = T_camera_to_marker[:3,3] + t * ray_marker

    return p_marker[:2]

def convert(raw_data):
    converted = []
    for area in raw_data["areas"]:
        T = area["T"]
        if T.size == 0:
            continue

        marker_data = []
        for marker in area["markers"]:
            converted_pos = convertToMarkerCoords(marker["position"].tolist(), T)
            marker_data.append({
                "id": marker["id"],
                "position": converted_pos
            })
        house_data = []
        for house in area["houses"]:
            converted_points = []
            for point in house["points"]:
                converted_points.append(convertToMarkerCoords(point, T))
            house_data.append({
                "points": converted_points,
                "class": house["class"],
                "confidence": house["confidence"]
            })

        path_data = []
        for path in area["paths"]:
            converted_points = []
            for point in path["points"]:
                converted_points.append(convertToMarkerCoords(point, T))
            path_data.append({
                "points": converted_points
            })

        converted.append({
            "markers": marker_data,
            "houses": house_data,
            "paths": path_data
        })
    return converted

