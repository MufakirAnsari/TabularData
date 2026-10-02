import os
from pathlib import Path

slurm_dir = Path("slurm")
for filepath in slurm_dir.glob("*.sh"):
    with open(filepath, "rb") as f:
        content = f.read()
    content = content.replace(b"\r\n", b"\n")
    with open(filepath, "wb") as f:
        f.write(content)
print("Converted all .sh files to Unix LF line endings.")
