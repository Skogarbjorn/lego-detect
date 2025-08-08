import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_size = 200 

for i in range(2):
    marker_image = aruco.generateImageMarker(aruco_dict, i, marker_size)

    cv2.imwrite(f"aruco_marker_{i}.png", marker_image)
