def get_camera_settings_for_current_view(viewer):
    """Capture current camera view settings"""

    return {
        "distance": viewer.cam.distance,
        "azimuth": viewer.cam.azimuth,
        "elevation": viewer.cam.elevation,
        "lookat": viewer.cam.lookat.copy(),
    }
