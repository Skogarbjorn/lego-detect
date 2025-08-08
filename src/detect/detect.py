import json
import numpy as np
from detect.house_detect import detect_houses, draw_houses, CLASS_NAMES
from detect.area_detect import detect_area, draw_area
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_JSON = os.path.join(CURRENT_DIR, "..", "..", "output", "raw.json")
camera_calibration_file = os.path.join(CURRENT_DIR, "..", "..", "misc", "camera_calibration.npz")

data = np.load(camera_calibration_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

class Detector:
    def __init__(self):
        self.annotated_houses = None

    def convertToMarkerCoords(self, point, T_camera_to_marker):
        x,y = point
        x_norm = (x - camera_matrix[0,2]) / camera_matrix[0,0]
        y_norm = (y - camera_matrix[1,2]) / camera_matrix[1,1]

        ray_camera = np.array([x_norm, y_norm, 1.0])

        ray_marker = T_camera_to_marker[:3, :3] @ ray_camera

        t = -T_camera_to_marker[2,3] / ray_marker[2]
        p_marker = T_camera_to_marker[:3,3] + t * ray_marker

        return p_marker

    def detect(self, frame):
        rvec, tvec, rectangle_3d, T_camera_to_marker = detect_area(frame, camera_matrix, dist_coeffs)
        boxes, confidences, indices, class_ids = detect_houses(frame)

        if len(indices) != 0:
            frame = draw_houses(frame, boxes, confidences, indices, class_ids)
        if rvec is None:
            return frame

        export = True

        house_points_marker = []
        if self.annotated_houses is not None:
            for house in self.annotated_houses:
                points = house['points']
                name = house['name']

                p_markers = [self.convertToMarkerCoords(point, T_camera_to_marker) for point in points]

                house_points_marker.append((p_markers, name))
        elif len(indices) != 0:
            for index in indices:
                index = index[0] if isinstance(index, (tuple, list, np.ndarray)) else index
                x, y, w, h = boxes[index]
                points = [
                    (x, y),                  
                    (x + w, y),              
                    (x, y + h),              
                    (x + w, y + h)           
                ]

                p_markers = [self.convertToMarkerCoords(point, T_camera_to_marker) for point in points]
                class_name = CLASS_NAMES[class_ids[index]]

                house_points_marker.append((p_markers, class_name))
        else:
            export = False

        if export:
            export_data = {
                    "houses": [],
                    "area": []
                    }

            for points, label in house_points_marker:
                export_data["houses"].append({
                    "class": label,
                    "corners": [point.tolist() for point in points]
                    })

            export_data["area"] = rectangle_3d.tolist()

            with open(RAW_JSON, "w") as f:
                json.dump(export_data, f, indent=2)

        frame = draw_area(frame, rvec, tvec, rectangle_3d, camera_matrix, dist_coeffs)
        return frame

    def update_shapes(self, shapes):
        self.annotated_houses = shapes

if __name__ == "__main__":
    print("todo!")
