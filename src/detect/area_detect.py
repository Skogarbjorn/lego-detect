import cv2
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

markerLength = 0.05
half_len = markerLength / 2.0

def detect_area(frame, camera_matrix, dist_coeffs):
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.flatten()
        paired = list(zip(ids, corners))
        paired.sort(key=lambda x: x[0])
        _sorted_ids, sorted_corners = zip(*paired)
        corners = list(sorted_corners)

    objPoints = np.array([
        [[-half_len,  half_len, 0.0]],
        [[ half_len,  half_len, 0.0]],
        [[ half_len, -half_len, 0.0]],
        [[-half_len, -half_len, 0.0]]
        ], dtype=np.float32)

    rvecs = [None]*len(corners)
    tvecs = [None]*len(corners)

    for i in range(len(corners)):
        _, rvecs[i], tvecs[i] = cv2.solvePnP(
                objPoints, 
                corners[i], 
                camera_matrix,
                dist_coeffs)

    if len(corners) >= 2:
        R1, _ = cv2.Rodrigues(rvecs[0])
        t1 = tvecs[0].reshape(3,1)

        T_marker1_to_camera = np.eye(4)
        T_marker1_to_camera[:3, :3] = R1
        T_marker1_to_camera[:3, 3] = t1[:, 0]
        T_camera_to_marker1 = np.linalg.inv(T_marker1_to_camera)

        R2, _ = cv2.Rodrigues(rvecs[1])
        t2 = tvecs[1].reshape(3,1)

        T_marker2_to_camera = np.eye(4)
        T_marker2_to_camera[:3, :3] = R2
        T_marker2_to_camera[:3, 3] = t2[:, 0]

        T_marker2_in_marker1 = T_camera_to_marker1 @ T_marker2_to_camera

        pos_in_marker1 = T_marker2_in_marker1[:3, 3]

        pA = np.array([0,0,0], dtype=np.float32)
        pB = pos_in_marker1

        p0 = np.array([pA[0], pA[1], 0])
        p1 = np.array([pB[0], pA[1], 0])
        p2 = np.array([pB[0], pB[1], 0])
        p3 = np.array([pA[0], pB[1], 0])

        rectangle_3d = np.array([p0, p1, p2, p3], dtype=np.float32)

        return (rvecs[0], tvecs[0], rectangle_3d, T_camera_to_marker1)
    return (None, None, None, None)

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

        rvec, tvec, rectangle_3d, T_camera_to_marker = detect_area(frame, camera_matrix, dist_coeffs)
        drawn_frame = draw_area(frame, rvec, tvec, rectangle_3d, camera_matrix, dist_coeffs)

        cv2.imshow("area detect", drawn_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabber.release()
    cv2.destroyAllWindows()
