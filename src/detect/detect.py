import json
import cv2
from lib.frame_grabber import FrameGrabber
from scipy.spatial.transform import Rotation
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#RAW_JSON = os.path.join(CURRENT_DIR, "..", "..", "output", "raw.json")

HISTORY_LENGTH = 5

camera_calibration_file = os.path.join(CURRENT_DIR, "..", "..", "misc", "camera_calibration.npz")

data = np.load(camera_calibration_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

class Detector:
    def __init__(self, num_cameras):
        self.num_cameras = num_cameras

        self.T_history = []
        self.markers_history = []
        self.houses_history = []
        self.paths_history = []

    def export(self):
        avg_T = self.export_T()
        avg_markers = self.export_markers()
        avg_houses = self.export_houses()
        avg_paths = self.export_paths()

        export_data = {
            "areas": []
        }

        for i in range(self.num_cameras):
            export_data["areas"].append({
                "T": avg_T[i] if i < len(avg_T) else [],
                "markers": avg_markers[i] if i < len(avg_markers) else [],
                "houses": avg_houses[i] if i < len(avg_houses) else [],
                "paths": avg_paths[i] if i < len(avg_paths) else []
            })

        return export_data

    def export_paths(self):
        gamer = [self.average_paths(instance) for instance in zip(*self.paths_history)]
        return gamer
    def export_T(self):
        return [self.average_T(instance) for instance in zip(*self.T_history) if any(x is not None for x in instance)]
    def export_markers(self):
        return [self.average_markers(instance) for instance in zip(*self.markers_history)]
    def export_houses(self):
        return [self.average_houses(instance) for instance in zip(*self.houses_history)]

    def detect_paths(self, frames):
        paths = []
        for frame in frames:
            if frame is None:
                continue
            path_data = []
            boxes = detect_paths(frame)
            
            if len(boxes) != 0:
                for box in boxes:
                    path_data.append({
                        "points": box
                    })

            paths.append(path_data)

        if len(self.paths_history) >= HISTORY_LENGTH:
            self.paths_history.pop(0)

        self.paths_history.append(paths)

        return self.export_paths()

    def detect(self, frames):
        Ts = []
        markers = []
        houses = []

        self.detect_paths(frames)

        for frame in frames:
            print(frame)
            if frame is None:
                continue
            ids, positions, T = detect_area(frame, camera_matrix, dist_coeffs)
            boxes, confidences, class_ids = detect_houses(frame)

            marker_data = []
            if ids is not None and T is not None:
                for id in ids:
                    marker_data.append({
                        "id": int(id),
                        "position": [pos.tolist() for pos in positions[id]]
                    })


            house_data = []
            if len(boxes) != 0:
                for i, box in enumerate(boxes):
                    points = box.tolist()
                    class_name = CLASS_NAMES[class_ids[i]]
                    confidence = confidences[i]

                    house_data.append({
                        "points": points, 
                        "class": class_name, 
                        "confidence": float(confidence)
                    })

            Ts.append(T)
            markers.append(marker_data)
            houses.append(house_data)

        if len(self.T_history) >= HISTORY_LENGTH:
            self.T_history.pop(0)
        if len(self.markers_history) >= HISTORY_LENGTH:
            self.markers_history.pop(0)
        if len(self.houses_history) >= HISTORY_LENGTH:
            self.houses_history.pop(0)

        self.T_history.append(Ts)
        self.markers_history.append(markers)
        self.houses_history.append(houses)

        gamer = self.export()
        print(gamer)
        return gamer

    def update_shapes(self, shapes):
        self.annotated_houses = shapes

    def average_T(self, transforms):
        translations = np.array([t[:3, 3] for t in transforms if t is not None])
        avg_translation = np.mean(translations, axis=0)

        rotations = [Rotation.from_matrix(t[:3, :3]) for t in transforms if t is not None]
        quats = np.array([r.as_quat() for r in rotations])  

        avg_quat = np.mean(quats, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)

        avg_rotation = Rotation.from_quat(avg_quat).as_matrix()

        avg_transform = np.eye(4)
        avg_transform[:3, :3] = avg_rotation
        avg_transform[:3, 3] = avg_translation

        return avg_transform

    def average_paths(self, history):
        hits = []
        for paths in history:
            for path in paths:
                hit = False
                for i, targets in enumerate(hits):
                    if self._paths_overlap(path, targets[-1]):
                        hits[i].append(path)
                        hit = True
                if not hit:
                    hits.append([path])
        return [hit[-1] for hit in hits if len(hit) >= 2]

    def average_houses(self, history):
        hits = []
        for houses in history:
            for house in houses:
                hit = False
                for i, targets in enumerate(hits):
                    if self._inside(house, targets[-1]):
                        hits[i].append(house)
                        hit = True
                if not hit:
                    hits.append([house])
        return [hit[-1] for hit in hits if len(hit) >= len(history) // 2]

    def average_markers(self, history):
        grouped = {}
        for markers in history:
            for marker in markers:
                id = marker["id"]
                if id not in grouped:
                    grouped[id] = []
                grouped[id].append(marker)

        #counts = {k: len(v) for k, v in data.items()}
        #max_count = max(counts.values())
        #threshold = max_count * 0.5

        #filtered = {k: v for k, v in data.items() if len(v) >= threshold}

        averaged = []
        for k, positions_list in grouped.items():
            positions_array = np.array([p["position"] for p in positions_list])
            mean_pos = positions_array.mean(axis=0)
            averaged.append({"id": k, "position": mean_pos})
        return averaged


    def _inside(self, curr, target):
        center = np.mean(np.array(curr["points"], dtype=float), axis=0)
        return cv2.pointPolygonTest(np.array(target["points"], dtype=np.float32), center, False) >= 0

    def _get_path_properties(self, path):
        center = np.mean(path, axis=0)

        # Calculate edge vectors
        edges = path[1:] - path[:-1]
        edges = np.vstack((edges, path[0] - path[-1]))

        # Find the two unique edge lengths (for rectangle)
        edge_lengths = np.linalg.norm(edges, axis=1)
        unique_lengths = np.unique(edge_lengths)

        # Get min dimension (smaller side of rectangle)
        min_dim = np.min(unique_lengths)

        # Calculate rotation (angle of first edge)
        rotation = np.arctan2(edges[0][1], edges[0][0])

        return center, min_dim, rotation

    def _paths_overlap(self, path1, path2, position_threshold=0.8, rotation_threshold=0.2):
        """
        Check if two rectangles overlap based on:
            1. Center distance within 20% of min dimension
        2. Rotation difference within 20% margin
        """
        # Get properties for both rectangles
        center1, min_dim1, rot1 = self._get_path_properties(path1["points"])
        center2, min_dim2, rot2 = self._get_path_properties(path2["points"])

        # Use average min dimension as reference size
        avg_min_dim = (min_dim1 + min_dim2) / 2

        # Check position similarity
        distance = np.linalg.norm(center1 - center2)
        position_ok = distance < (position_threshold * avg_min_dim)

        # Check rotation similarity (handle angle wrap-around)
        angle_diff = np.abs(rot1 - rot2)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Smallest angle difference
        rotation_ok = angle_diff < (rotation_threshold * np.pi)

        return position_ok and rotation_ok


if __name__ == "__main__":
    from house_detect import detect_houses, CLASS_NAMES
    from area_detect import detect_area
    from path_detect import detect_paths
    grabber = FrameGrabber()
    detector = Detector()
    while True:
        frame = grabber.read()
        if frame is None:
            continue

        frames = detector.detect([frame])

        cv2.imshow("gamer", frames[0])
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    detector.export()

    cv2.destroyAllWindows()
else:
    from detect.house_detect import detect_houses, CLASS_NAMES
    from detect.area_detect import detect_area
    from detect.path_detect import detect_paths
