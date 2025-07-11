
import numpy as np
import json


if __name__=='__main__':
    radius      = 1.7 * 0.8
    theta       = 80
    phi_range   = (0, 75)
    size        = 25

    cameras = []
    for phi in np.linspace(phi_range[0], phi_range[1], size):
        loc = [
            radius * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
            radius * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
            radius * np.cos(np.deg2rad(theta)),
        ]

        cameras.append({
            "location": loc,
            "rotation": [theta, 0.0, 180 - phi],
            "probability": 1.0
        })

    with open('camera.json', 'w') as f:
        json.dump(cameras, f, indent=4)