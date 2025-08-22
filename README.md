# Setup

## Camera Calibration

This is pretty insignificant, and I think the program is completely functional without recalibrating for your camera, but it just involves taking a few pictures from the camera you intend to use of a [checkerboard pattern](https://markhedleyjones.com/projects/calibration-checkerboard-collection), and putting those into misc/calibration_images and running lib/calibrate_camera

## Aruco Markers

Since the detection requires a coordinate field, each camera needs to see two ArUco markers, so printing at least two of those out is required. They can be generated using lib/aruco_gen. The physical size of the printouts should also be measured and set as `markerLength` in detect/area_detect to provide consistent real-world coordinates, although not necessary.

## Dependencies

Everything is set up in a python project so `pip install -e .` should install all required modules

# Running

`visualization.py` is the entry point and should be called with the links to camera feeds as arguments, i.e. 0 for `/dev/video0` or an ip string if sourcing from an online feed.

For 2 local cameras, you would call `python -m visualization 0 1` if the cameras are mounted as `/dev/video0` and `/dev/video1`.
(i dont know how cameras work on windows, so refer to the opencv documentation for `cv2.VideoCapture`)

Currently, at least one camera has to detect two ArUco markers within the first second, but this should be fixed soon


