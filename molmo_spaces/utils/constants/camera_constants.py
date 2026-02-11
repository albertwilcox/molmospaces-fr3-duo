"""Camera hardware constants for fisheye warping and image processing."""

# GoPro Hero 11 Black specifications (wide FOV mode)
GOPRO_VERTICAL_FOV = 139.0  # degrees
GOPRO_CAMERA_WIDTH = 640
GOPRO_CAMERA_HEIGHT = 480

# Model output dimensions (4:3 aspect ratio)
MODEL_43_WIDTH = 320
MODEL_43_HEIGHT = 240

# Fisheye distortion parameters
# Default parameters calibrated for GoPro fisheye distortion
DEFAULT_DISTORTION_PARAMETERS = {
    "k1": 0.051,
    "k2": 0.144,
    "k3": 0.015,
    "k4": -0.018,
}

NULL_DISTORTION_PARAMETERS = {
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
    "k4": 0.0,
}

# Unity-based distortion parameters (Alvaro's calibration)
ALVARO_UNITY_DISTORTION_PARAMETERS = {
    "zoomPercent": 0.49,
    "k1": 0.9,
    "k2": 5.2,
    "k3": -13.0,
    "k4": 16.3,
    "intensityX": 1.0,
    "intensityY": 0.98,
}

ALVARO_UNITY_DISTORTION_PARAMETER_RANGES = {
    "zoomPercent": (0.45, 0.53),
    "k1": (0.8, 1.0),
    "k2": (5.0, 5.4),
    "k3": (-14.0, -12.0),
    "k4": (15.0, 17.0),
    "intensityX": (0.95, 1.05),
    "intensityY": (0.93, 1.03),
}

# Default crop percentage to remove edge distortion
DEFAULT_CROP_PERCENT = 0.30
