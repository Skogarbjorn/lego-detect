# Setup

## Camera Calibration

This is pretty insignificant, and I think the program is completely functional without recalibrating for your camera, but it just involves taking a few pictures from the camera you intend to use of a [checkerboard pattern](https://markhedleyjones.com/projects/calibration-checkerboard-collection), and putting those into misc/calibration_images and running lib/calibrate_camera

## Aruco Markers

Since the detection requires a coordinate field, each camera needs to see two ArUco markers, so printing at least two of those out is required. They can be generated using lib/aruco_gen.

## Dependencies

Everything is set up in a python project so `pip install -e .` should install all required modules

# Running

`visualization.py` is the entry point should be called with the links to camera feeds as arguments, i.e. 0 for `/dev/video0` or an ip string if sourcing from an online feed.

For 2 local cameras, you would call `python -m visualization 0 1` if the cameras are mounted as `/dev/video0` and `/dev/video1`.

Currently, at least one camera has to detect two ArUco markers within the first second, but this should be fixed soon


