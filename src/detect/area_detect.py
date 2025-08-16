import cv2
import numpy as np
import os
import cv2.aruco as aruco
import numpy as np

from lib.frame_grabber import FrameGrabber

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementWinSize = 5
parameters.cornerRefinementMaxIterations = 100
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

markerLength = 6
half_len = markerLength / 2.0

import numpy as np
import cv2

def detect_area(frame, camera_matrix, dist_coeffs):
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None or len(ids) == 0:
        return None, None, None

    ids = ids.flatten()
    paired = list(zip(ids, corners))
    paired.sort(key=lambda x: x[0])
    sorted_ids, sorted_corners = zip(*paired)
    ids = list(sorted_ids)
    corners = list(sorted_corners)

    objPoints = np.array([
        [[-half_len,  half_len, 0.0]],
        [[ half_len,  half_len, 0.0]],
        [[ half_len, -half_len, 0.0]],
        [[-half_len, -half_len, 0.0]]
    ], dtype=np.float32)

    positions = {}
    T_marker_to_camera = None

    for i in range(len(corners)):
        _, rvec, tvec = cv2.solvePnP(
            objPoints,
            corners[i],
            camera_matrix,
            dist_coeffs
        )

        img_pts, _ = cv2.projectPoints(
            np.array([[0, 0, 0]], dtype=np.float32),  
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        positions[ids[i]] = tuple(img_pts[0].ravel())

        if i == 0:
            R, _ = cv2.Rodrigues(rvec)
            T_marker_to_camera = np.eye(4, dtype=np.float32)
            T_marker_to_camera[:3, :3] = R
            T_marker_to_camera[:3, 3] = tvec.flatten()

    return ids, positions, np.linalg.inv(T_marker_to_camera)

def draw_area(frame, rvec, tvec, rectangle_3d, camera_matrix, dist_coeffs):
    if rvec is None or tvec is None or rectangle_3d is None:
        return frame
    imgpts, _ = cv2.projectPoints(rectangle_3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    cv2.polylines(frame, [imgpts], isClosed=True, color=(0,255,0), thickness=2)
    return frame

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
camera_calibration_file = os.path.join(CURRENT_DIR, "..", "..", "misc", "camera_calibration.npz")

if __name__ == "__main__":
    grabber = FrameGrabber()

    data = np.load(camera_calibration_file)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    while True:
        frame = grabber.read()
        if frame is None:
            continue

        detect_area(frame, camera_matrix, dist_coeffs)

        cv2.imshow("area detect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabber.release()
    cv2.destroyAllWindows()
