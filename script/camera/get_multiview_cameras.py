import os
import math
import numpy as np
import json


if __name__=='__main__':
    # 0. configuration
    radius = 4
    elevation_deg = 60
    fov_deg = 45
    n = 4

    output_dir = "outputs"

    # 1. multiview cameras
    cameras = []
    for azimuth_deg in np.linspace(0, 360, n, endpoint=False):
        elevation   = math.radians(elevation_deg-90)
        azimuth     = math.radians(azimuth_deg+90)
        
        location = [
            - radius * math.cos(elevation) * math.cos(azimuth),
            - radius * math.cos(elevation) * math.sin(azimuth),
            - radius * math.sin(elevation)
        ]
        cameras.append({
            "location": location,
            "rotation": [elevation_deg, 0.0, azimuth_deg],
            "fov": fov_deg,
            "probability": 1.0
        })

    # 2. output
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"mv_cameras_elev{elevation_deg:d}_n{n}.json"), "w") as f:
        json.dump(cameras, f, indent=4)
        
    print("DONE")