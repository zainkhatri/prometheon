"""Unzip all takeout zips into /srv/mergerfs/PROMETHEUS/PHOTOS/."""
import subprocess
import glob
import os

PHOTOS = "/srv/mergerfs/PROMETHEUS/PHOTOS"
zips = sorted(glob.glob(os.path.join(PHOTOS, "takeout-*.zip")))
print(f"Found {len(zips)} zip files")
for z in zips:
    size_gb = os.path.getsize(z) / (1024**3)
    print(f"Unzipping {os.path.basename(z)} ({size_gb:.1f} GB) ...", flush=True)
    r = subprocess.run(["unzip", "-o", "-q", z, "-d", PHOTOS], timeout=7200)
    if r.returncode == 0:
        print(f"  Done: {os.path.basename(z)}", flush=True)
    else:
        print(f"  Error (exit {r.returncode})", flush=True)
print("All done.", flush=True)
