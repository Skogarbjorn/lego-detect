import os
import cv2
import numpy as np
import glob

chessboard_size = (9, 6)
square_size = 0.025  
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []
image_size = None  

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(CURRENT_DIR, "..", "..", "misc", "calibration_images", "*.png")
calibration_path = os.path.join(CURRENT_DIR, "..", "..", "misc", "camera_calibration.npz")

images = glob.glob(images_path)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        if image_size is None:
            image_size = gray.shape[::-1]  

if not objpoints:
    raise RuntimeError("found no corners in any images?!?")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

np.savez(calibration_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
