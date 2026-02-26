"""Recycling bin for PROMETHEON. Moves files to trash instead of deleting them."""

import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

TRASH_DIR = Path("/Volumes/PROMETHEUS/.prometheon-trash")
TRASH_DIR.mkdir(parents=True, exist_ok=True)


def _meta_path(trash_name: str) -> Path:
    return TRASH_DIR / f"{trash_name}.meta.json"


def trash_file(file_path: str) -> dict:
    """Move a file/directory to the recycling bin with metadata."""
    src = Path(file_path).resolve()
    if not src.exists():
        return {"success": False, "error": f"Path does not exist: {file_path}"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_name = f"{timestamp}_{src.name}"
    dest = TRASH_DIR / trash_name

    try:
        shutil.move(str(src), str(dest))
    except Exception as e:
        return {"success": False, "error": f"Failed to move to trash: {e}"}

    meta = {
        "original_path": str(src),
        "trash_name": trash_name,
        "trashed_at": datetime.now().isoformat(),
        "size": dest.stat().st_size if dest.is_file() else _dir_size(dest),
    }
    _meta_path(trash_name).write_text(json.dumps(meta, indent=2))

    return {"success": True, "message": f"Moved to trash: {src.name}", "trash_name": trash_name}


def list_trash() -> list[dict]:
    """List all items in the recycling bin."""
    items = []
    for meta_file in sorted(TRASH_DIR.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text())
            age = datetime.now() - datetime.fromisoformat(meta["trashed_at"])
            meta["age_days"] = age.days
            meta["age_str"] = f"{age.days}d {age.seconds // 3600}h ago"
            meta["purge_in"] = max(0, 30 - age.days)
            items.append(meta)
        except Exception:
            continue
    return items


def restore(trash_name: str) -> dict:
    """Restore an item from the recycling bin to its original location."""
    item_path = TRASH_DIR / trash_name
    meta_file = _meta_path(trash_name)

    if not item_path.exists():
        return {"success": False, "error": f"Item not found in trash: {trash_name}"}

    if not meta_file.exists():
        return {"success": False, "error": f"Metadata not found for: {trash_name}"}

    meta = json.loads(meta_file.read_text())
    original = Path(meta["original_path"])

    if original.exists():
        return {"success": False, "error": f"Original path already exists: {original}"}

    original.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(item_path), str(original))
        meta_file.unlink()
        return {"success": True, "message": f"Restored to: {original}"}
    except Exception as e:
        return {"success": False, "error": f"Restore failed: {e}"}


def purge_old(days: int = 30) -> dict:
    """Permanently delete items older than `days` days."""
    purged = []
    for meta_file in TRASH_DIR.glob("*.meta.json"):
        try:
            meta = json.loads(meta_file.read_text())
            trashed_at = datetime.fromisoformat(meta["trashed_at"])
            if datetime.now() - trashed_at > timedelta(days=days):
                trash_name = meta["trash_name"]
                item_path = TRASH_DIR / trash_name
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                elif item_path.exists():
                    item_path.unlink()
                meta_file.unlink()
                purged.append(trash_name)
        except Exception:
            continue
    return {"purged_count": len(purged), "purged": purged}


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "purge":
        result = purge_old()
        print(f"Purged {result['purged_count']} items older than 30 days.")
