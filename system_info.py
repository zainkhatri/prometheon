"""System info gathering for PROMETHEON — runs locally on the NAS or Mac."""

import json
import os
import platform
import subprocess
import sys
import time
import psutil
from datetime import timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_CACHE_FILE = os.path.join(SCRIPT_DIR, ".folder_sizes.json")
DISK_CACHE_FILE = os.path.join(SCRIPT_DIR, ".disk_stats.json")
NAS_DRIVES_FILE = os.path.join(SCRIPT_DIR, ".nas_drives.json")
DISK_CACHE_TTL = 60  # seconds

IS_MAC = sys.platform == "darwin"

# ─── Pool root ───
# On the NAS (Linux): /srv/mergerfs/PROMETHEUS
# On Mac (dev/SMB):   /Volumes/PROMETHEUS
if IS_MAC:
    POOL_ROOT = "/Volumes/PROMETHEUS"
else:
    POOL_ROOT = "/srv/mergerfs/PROMETHEUS"

# ─── Drive detection ───
# Linux NAS: match by device path fragment
LINUX_DRIVE_MAP = {
    "nvme0n1": "EVO",
    "sda": "AirDisk",
    "sdb": "T7",
    "sdc": "T9",
    "sdd": "T5",
}

# macOS: match by volume name
MAC_VOLUME_MAP = {
    "PROMETHEUS": "PROMETHEUS",
    "T5": "T5",
    "T7": "T7",
    "EVO": "EVO",
    "T9": "T9",
    "Samsung_T5": "T5",
    "Samsung_T7": "T7",
    "Samsung_T9": "T9",
}


def _format_bytes(b: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PiB"


def _get_disks():
    """Get disk info with time-based caching. Works on Linux NAS and macOS."""
    # Serve from cache if fresh enough
    try:
        with open(DISK_CACHE_FILE) as f:
            cache = json.load(f)
        if time.time() - cache.get("ts", 0) < DISK_CACHE_TTL:
            return cache["disks"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    if IS_MAC:
        disks = _get_disks_mac()
    else:
        disks = _get_disks_linux()

    # Persist cache
    try:
        with open(DISK_CACHE_FILE, "w") as f:
            json.dump({"ts": time.time(), "disks": disks}, f)
    except OSError:
        pass

    return disks


def _physical_disk_bytes(dev_name: str) -> int:
    """Get full physical disk capacity from /sys/block (not just the mounted partition)."""
    try:
        with open(f'/sys/block/{dev_name}/size') as f:
            return int(f.read().strip()) * 512  # sectors → bytes
    except (OSError, ValueError):
        return 0


def _make_disk_entry(mount, name, total_override: int = 0):
    """Create a disk info dict from a mount point. total_override replaces partition total with physical disk size."""
    try:
        usage = psutil.disk_usage(mount)
    except (OSError, PermissionError):
        return None
    total = total_override if total_override else usage.total
    return {
        "mount": mount,
        "name": name,
        "total": _format_bytes(total),
        "used": _format_bytes(usage.used),
        "free": _format_bytes(total - usage.used),
        "percent": round(usage.used / total * 100, 1) if total else 0,
    }


def _get_disks_mac():
    """On Mac: get PROMETHEUS pool locally, read individual drive stats from shared NAS file."""
    disks = []

    # PROMETHEUS pool — visible via SMB
    entry = _make_disk_entry("/Volumes/PROMETHEUS", "PROMETHEUS")
    if entry:
        disks.append(entry)

    # Individual NAS drives — read from .nas_drives.json (written by the NAS service)
    try:
        with open(NAS_DRIVES_FILE) as f:
            nas_data = json.load(f)
        for d in nas_data.get("drives", []):
            disks.append({
                "mount": d.get("device", ""),
                "name": d["name"],
                "total": d.get("total", "—"),
                "used": d.get("used", "—"),
                "free": d.get("free", "—"),
                "percent": d.get("percent", 0),
            })
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    return disks


def _get_disks_linux():
    """Detect drives on the Linux NAS and write results to shared file."""
    disks = []
    seen_names = set()

    # PROMETHEUS merged pool
    entry = _make_disk_entry(POOL_ROOT, "PROMETHEUS")
    if entry:
        disks.append(entry)
        seen_names.add("PROMETHEUS")

    # Individual drives by device path (also written to shared file for Mac clients)
    drive_entries = []
    for part in psutil.disk_partitions(all=True):
        mp = part.mountpoint
        dev = part.device

        name = None
        dev_frag = None
        for frag, drive_name in LINUX_DRIVE_MAP.items():
            if frag in dev:
                name = drive_name
                dev_frag = frag
                break

        if not name:
            if mp.startswith("/srv/dev-disk-by"):
                slug = mp.rsplit("-", 1)[-1] if "-" in mp else mp.split("/")[-1]
                name = slug[:8]
            else:
                continue

        if name in seen_names:
            continue

        phys = _physical_disk_bytes(dev_frag) if dev_frag else 0
        entry = _make_disk_entry(mp, name, total_override=phys)
        if entry:
            seen_names.add(name)
            disks.append(entry)
            drive_entries.append(entry)

    # Write individual drive stats to shared file so Mac clients can read them
    try:
        with open(NAS_DRIVES_FILE, "w") as f:
            json.dump({"ts": time.time(), "drives": drive_entries}, f)
    except OSError:
        pass

    return disks


_folder_cache = {"data": None}


def _load_folder_cache():
    try:
        with open(FOLDER_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_folder_cache(cache):
    with open(FOLDER_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _du_single(path):
    """Get size of a single directory using du."""
    try:
        # macOS du doesn't support --block-size
        if IS_MAC:
            result = subprocess.run(
                ["du", "-sk", path],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return int(result.stdout.split("\t")[0]) * 1024
        else:
            result = subprocess.run(
                ["du", "-s", "--block-size=1", path],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return int(result.stdout.split("\t")[0])
    except Exception:
        pass
    return None


def _get_folder_sizes():
    """Get sizes of top 3 largest folders. Only re-measures folders whose mtime changed."""
    # Skip on Mac — du over SMB is painfully slow
    if IS_MAC:
        return _folder_cache.get("data") or []

    disk_cache = _load_folder_cache()

    try:
        subdirs = [d for d in os.listdir(POOL_ROOT)
                    if os.path.isdir(os.path.join(POOL_ROOT, d))]
    except OSError:
        return _folder_cache.get("data") or []

    changed = False
    for name in subdirs:
        path = os.path.join(POOL_ROOT, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue

        cached = disk_cache.get(name)
        if cached and cached.get("mtime") == mtime:
            continue

        size = _du_single(path)
        if size is not None:
            disk_cache[name] = {"size": size, "mtime": mtime}
            changed = True

    for name in list(disk_cache):
        if name not in subdirs:
            del disk_cache[name]
            changed = True

    if changed:
        _save_folder_cache(disk_cache)

    folders = []
    for name, info in disk_cache.items():
        folders.append({"name": name, "size": info["size"], "display": _format_bytes(info["size"])})
    folders.sort(key=lambda x: x["size"], reverse=True)
    top3 = folders[:3]

    try:
        usage = psutil.disk_usage(POOL_ROOT)
        for f in top3:
            f["percent"] = round(f["size"] / usage.total * 100, 1)
    except Exception:
        for f in top3:
            f["percent"] = 0

    _folder_cache["data"] = top3
    return top3


def _get_cpu_temp():
    """Read CPU temperature."""
    if IS_MAC:
        # macOS: try powermetrics or osx-cpu-temp if available
        try:
            result = subprocess.run(
                ["osx-cpu-temp"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip().replace("°C", "")
                return round(float(temp_str), 1)
        except Exception:
            pass
        return None

    # Linux: hwmon
    hwmon_base = "/sys/class/hwmon"
    try:
        for hwmon in os.listdir(hwmon_base):
            name_path = os.path.join(hwmon_base, hwmon, "name")
            try:
                with open(name_path) as f:
                    name = f.read().strip()
            except OSError:
                continue
            if name in ("k10temp", "coretemp"):
                temp_path = os.path.join(hwmon_base, hwmon, "temp1_input")
                try:
                    with open(temp_path) as f:
                        return round(int(f.read().strip()) / 1000, 1)
                except (OSError, ValueError):
                    continue
    except OSError:
        pass
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000, 1)
    except Exception:
        return None


def get_system_info() -> dict:
    """Gather system information for display."""
    # OS info
    if IS_MAC:
        os_name = f"macOS {platform.mac_ver()[0]}"
    else:
        try:
            with open("/etc/os-release") as f:
                os_info = {}
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        os_info[k] = v.strip('"')
            os_name = os_info.get("PRETTY_NAME", "Linux")
        except Exception:
            os_name = "Linux"

    # Uptime
    try:
        uptime_secs = psutil.boot_time()
        delta = timedelta(seconds=int(time.time() - uptime_secs))
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes = remainder // 60
        parts = []
        if days: parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours: parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes: parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        uptime_str = ", ".join(parts) if parts else "just started"
    except Exception:
        uptime_str = "N/A"

    # CPU
    cpu_percent = psutil.cpu_percent(interval=0)
    if IS_MAC:
        cpu_desc = f"{platform.processor()} ({psutil.cpu_count()} cores)"
    else:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_desc = line.split(":")[1].strip()
                        break
                else:
                    cpu_desc = f"{psutil.cpu_count()} cores"
        except Exception:
            cpu_desc = f"{psutil.cpu_count()} cores"

    # Memory
    mem = psutil.virtual_memory()

    # Disks (cached)
    disks = _get_disks()

    # Top-level folder sizes
    folders = _get_folder_sizes()

    return {
        "hostname": "PROMETHEUS",
        "folders": folders,
        "os": os_name,
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "cpu": cpu_desc,
        "cpu_percent": cpu_percent,
        "cpu_temp": _get_cpu_temp(),
        "memory_total": _format_bytes(mem.total),
        "memory_used": _format_bytes(mem.used),
        "memory_percent": round(mem.percent, 1),
        "uptime": uptime_str,
        "disks": disks,
        "python": platform.python_version(),
    }
