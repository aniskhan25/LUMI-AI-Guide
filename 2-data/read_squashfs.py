"""
Read images from a SquashFS archive mounted at /data inside the container.
Run via run_squashfs.sh — do not invoke directly.
"""

from pathlib import Path
from PIL import Image

data_dir = Path("/data")
images = sorted(data_dir.glob("*.png"))

print(f"Found {len(images)} images in {data_dir}")
print(f"First 5: {[p.name for p in images[:5]]}")

# Verify images are readable
for p in images[:5]:
    img = Image.open(p)
    assert img.size == (64, 64), f"Unexpected size {img.size} for {p.name}"

print("SQUASHFS READ OK")
