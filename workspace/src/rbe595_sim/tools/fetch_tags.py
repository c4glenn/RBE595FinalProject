#!/usr/bin/env python3
"""Download tag36h11 AprilTag images and upscale them (nearest-neighbour) for use as textures.

Usage: fetch_tags.py [MAX_ID]   (default 11, one per spice bottle; writes tags/tag36_11_<id:05d>.png)
"""
import io
import subprocess
import sys
from pathlib import Path

from PIL import Image

URL = "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tag36h11/tag36_11_{:05d}.png"
SCALE = 16  # the source images are 10x10 px; Gazebo would otherwise blur them
OUT = Path(__file__).resolve().parent.parent / "tags"

max_id = int(sys.argv[1]) if len(sys.argv) > 1 else 11
OUT.mkdir(exist_ok=True)
for tag_id in range(max_id + 1):
    # curl rather than urllib: python.org builds on macOS often lack a CA bundle
    data = subprocess.run(
        ["curl", "-sSfL", URL.format(tag_id)], check=True, capture_output=True
    ).stdout
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST).save(
        OUT / f"tag36_11_{tag_id:05d}.png"
    )
