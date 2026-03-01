"""Scan PHOTOS directory and build photo_index.json.

Thumbnails are generated on-demand by app.py (lazy, disk-cached).
This script only extracts dates and builds the index — runs in seconds.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# NAS-local paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "PHOTOS", "PHOTOS")
THUMB_DIR = os.path.join(SCRIPT_DIR, "static", "thumbs")
THUMB_HQ_DIR = os.path.join(SCRIPT_DIR, "static", "thumbs_hq")
INDEX_FILE = os.path.join(SCRIPT_DIR, "photo_index.json")

WORKERS = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
ALL_EXTS = IMAGE_EXTS | VIDEO_EXTS


def hash_path(path):
    return hashlib.md5(path.encode()).hexdigest()


def infer_date_from_path(filepath):
    import calendar
    rel = os.path.relpath(filepath, PHOTOS_ROOT)
    parts = rel.split(os.sep)
    for i, part in enumerate(parts):
        if re.match(r"^(19|20)\d{2}$", part):
            year = int(part)
            if i + 1 < len(parts) and re.match(r"^(0[1-9]|1[0-2])$", parts[i + 1]):
                # Month subfolder present — use mid-month
                month = int(parts[i + 1])
                estimated = datetime(year, month, 15, 12, 0, 0)
            else:
                # No month subfolder — spread across the year using filename hash
                # so files get distinct, reproducible dates rather than all July 15
                fname = os.path.basename(filepath)
                h = int(hashlib.md5(fname.encode()).hexdigest()[:8], 16)
                days_in_year = 366 if calendar.isleap(year) else 365
                day_of_year = (h % days_in_year) + 1
                month_days = [31, 29 if calendar.isleap(year) else 28,
                              31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                cum = 0
                month, day = 7, 1
                for m, d in enumerate(month_days, 1):
                    if day_of_year <= cum + d:
                        month, day = m, day_of_year - cum
                        break
                    cum += d
                hour = (h >> 8) % 24
                minute = (h >> 16) % 60
                estimated = datetime(year, month, day, hour, minute, 0)
            now = datetime.now()
            if estimated > now:
                estimated = now
            return estimated.timestamp()
    return None


def get_media_date(filepath):
    """Extract real date from EXIF/metadata. Falls back to mtime."""
    try:
        result = subprocess.run(
            ["exiftool", "-DateTimeOriginal", "-CreateDate", "-MediaCreateDate",
             "-s3", "-d", "%Y:%m:%d %H:%M:%S", filepath],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line and line != "0000:00:00 00:00:00":
                    parsed = parse_date(line)
                    if parsed:
                        return parsed
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags=creation_time", filepath],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            val = data.get("format", {}).get("tags", {}).get("creation_time")
            if val:
                parsed = parse_date(val)
                if parsed:
                    return parsed
    except Exception:
        pass

    mtime = os.path.getmtime(filepath)
    date_from_path = infer_date_from_path(filepath)

    if date_from_path:
        path_year = datetime.fromtimestamp(date_from_path).year
        mtime_year = datetime.fromtimestamp(mtime).year
        # If the path says the file is from an earlier year than mtime, the file
        # was likely copied to the NAS recently — trust the path-based date.
        # If they agree on year, mtime is fine (Google Takeout sets mtime = shot date).
        if mtime_year > path_year:
            return date_from_path

    return mtime


def parse_date(date_str):
    date_str = date_str.strip()
    for fmt in [
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(date_str[:26], fmt)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def gen_thumb(filepath, out_path, size, quality, is_video):
    """Generate a single thumbnail.

    Uses PIL (+pillow-heif for HEIC) for images, ffmpeg for videos.
    """
    ext = os.path.splitext(filepath)[1].lower()

    # ── Videos: ffmpeg frame grab ──
    if is_video:
        try:
            for ss in ("1", "0"):
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-i", filepath, "-ss", ss, "-vframes", "1",
                     "-vf", f"scale={size}:-1", "-q:v", str(quality), out_path],
                    capture_output=True, timeout=30
                )
                if os.path.exists(out_path):
                    return True
            return False
        except Exception:
            return False

    # ── All images: PIL (pillow-heif registered so HEIC/HEIF open natively) ──
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    try:
        from PIL import Image, ImageOps
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size * 4), Image.LANCZOS)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.convert("RGB").save(out_path, "JPEG", quality=85, optimize=True)
        return os.path.exists(out_path)
    except Exception:
        return False


def _extract_date(job):
    """Extract date for a single file. Runs in worker process."""
    filepath, rel_path, ext = job
    is_video = ext in VIDEO_EXTS
    date = get_media_date(filepath)
    if date is None:
        date = os.path.getmtime(filepath)
    thumb_name = hash_path(rel_path) + ".jpg"
    return {
        "path": filepath,
        "thumb": f"/static/thumbs/{thumb_name}",
        "thumb_hq": f"/static/thumbs_hq/{thumb_name}",
        "date": date,
        "type": "video" if is_video else "image",
    }


def scan():
    print(f"Scanning {PHOTOS_ROOT} ...")
    print(f"Workers: {WORKERS}")
    start = time.time()

    SKIP_DIRS = {"takeouts"}

    SKIP_PATTERNS = {"branded", "low-res"}

    all_files = []
    for root, dirs, files in os.walk(PHOTOS_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if fname.startswith("._"):
                continue
            fname_lower = fname.lower()
            if any(pat in fname_lower for pat in SKIP_PATTERNS):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALL_EXTS:
                filepath = os.path.join(root, fname)
                all_files.append((filepath, ext))

    total = len(all_files)
    print(f"Found {total} media files in {time.time() - start:.1f}s")
    print(f"Extracting dates from {total} files...")

    jobs = []
    for filepath, ext in all_files:
        rel_path = os.path.relpath(filepath, PHOTOS_ROOT)
        jobs.append((filepath, rel_path, ext))

    entries = []
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        future_to_job = {pool.submit(_extract_date, job): job for job in jobs}
        for future in as_completed(future_to_job):
            try:
                result = future.result()
            except Exception:
                result = None
            if result:
                entries.append(result)
                done += 1
                if done % 2000 == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (total - done - failed) / rate if rate > 0 else 0
                    print(f"  {done}/{total} done ({rate:.0f}/s, ~{remaining/60:.1f}m remaining)")
            else:
                failed += 1

    entries.sort(key=lambda x: x["date"], reverse=True)

    with open(INDEX_FILE, "w") as f:
        json.dump(entries, f)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Indexed: {done}  Failed: {failed}")
    print(f"  -> {INDEX_FILE}")


def scan_incremental():
    """Fast incremental scan — only processes files not already in the index.

    Loads existing photo_index.json, finds new/missing files, processes only those,
    then merges and saves. Run this on a cron job for automatic pick-up of new photos.
    """
    print(f"[incremental] Scanning {PHOTOS_ROOT} for new files ...")
    start = time.time()

    SKIP_DIRS = {"takeouts"}

    # Load existing index
    existing = []
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE) as f:
                existing = json.load(f)
            print(f"[incremental] Loaded {len(existing)} existing entries.")
        except Exception as e:
            print(f"[incremental] Could not load existing index: {e} — doing full scan.")
            scan()
            return

    known_paths = {e["path"] for e in existing}

    SKIP_PATTERNS = {"branded", "low-res"}

    # Walk filesystem for all media files
    all_files = []
    for root, dirs, files in os.walk(PHOTOS_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if fname.startswith("._"):
                continue
            fname_lower = fname.lower()
            if any(pat in fname_lower for pat in SKIP_PATTERNS):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALL_EXTS:
                filepath = os.path.join(root, fname)
                if filepath not in known_paths:
                    all_files.append((filepath, ext))

    # Remove index entries for files that no longer exist on disk or match skip patterns
    def _should_keep(e):
        if not os.path.exists(e["path"]):
            return False
        fname_lower = os.path.basename(e["path"]).lower()
        return not any(pat in fname_lower for pat in SKIP_PATTERNS)

    still_exist = [e for e in existing if _should_keep(e)]
    removed = len(existing) - len(still_exist)
    if removed:
        print(f"[incremental] Removed {removed} entries for deleted files.")

    if not all_files:
        if removed:
            still_exist.sort(key=lambda x: x["date"], reverse=True)
            with open(INDEX_FILE, "w") as f:
                json.dump(still_exist, f)
            print(f"[incremental] Index updated (deletions only). Done in {time.time()-start:.1f}s")
        else:
            print(f"[incremental] No new files found. Done in {time.time()-start:.1f}s")
        return

    print(f"[incremental] Found {len(all_files)} new file(s) to index.")

    jobs = []
    for filepath, ext in all_files:
        rel_path = os.path.relpath(filepath, PHOTOS_ROOT)
        jobs.append((filepath, rel_path, ext))

    new_entries = []
    failed = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        future_to_job = {pool.submit(_extract_date, job): job for job in jobs}
        for future in as_completed(future_to_job):
            try:
                result = future.result()
            except Exception:
                result = None
            if result:
                new_entries.append(result)
            else:
                failed += 1

    merged = still_exist + new_entries
    merged.sort(key=lambda x: x["date"], reverse=True)

    with open(INDEX_FILE, "w") as f:
        json.dump(merged, f)

    elapsed = time.time() - start
    print(f"[incremental] Done in {elapsed:.1f}s — added {len(new_entries)}, removed {removed}, failed {failed}.")
    print(f"[incremental] Index now has {len(merged)} entries.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--incremental":
        scan_incremental()
    else:
        scan()
