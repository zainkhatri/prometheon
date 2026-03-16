#!/usr/bin/env python3
"""
Sync Goodnotes notebooks to PROMETHEON journals.

Reads the Goodnotes Mac app's local database (synced via iCloud) to build
a journal index. When Goodnotes Auto-Backup is enabled, copies exported
PDFs from iCloud Drive to the NAS journals directory.

Run periodically via launchd (see com.prometheon.goodnotes-sync.plist).
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

# Paths
GOODNOTES_DB = os.path.expanduser(
    "~/Library/Containers/com.goodnotesapp.x/Data/Library/Databases/projection.sqlite"
)
ICLOUD_DRIVE = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs"
)

# Detect NAS journals dir
if sys.platform == "darwin":
    JOURNALS_DIR = "/Volumes/PROMETHEUS/PERSONAL/journals"
else:
    JOURNALS_DIR = "/srv/mergerfs/PROMETHEUS/PERSONAL/journals"

META_FILE = os.path.join(JOURNALS_DIR, "goodnotes_meta.json")

# User's root folder ID (their personal notebook library)
USER_ROOT_FOLDER = "5A53E89E-F4C2-4548-8DD3-E9DF9FB4592E"

# Template/sticker document IDs to exclude
TEMPLATE_ROOT_FOLDERS = {
    "F6327919-7604-421F-9B60-C38A787F9F42",  # Templates
    "14AC2082-C07A-4C4F-AF22-A23ACC3B8A5F",  # Paper templates
}

# Known template names to skip
TEMPLATE_NAMES = {
    "Text Stamps", "Back To School", "Sticky Notes", "Everyday Stickers",
    "Mind Map Shapes", "Ruled Wide", "Squared Paper", "Ruled Narrow",
    "Bright", "Calligraphr-Template",
}


def read_goodnotes_db():
    """Read notebook metadata from Goodnotes projection database."""
    if not os.path.exists(GOODNOTES_DB):
        print(f"Goodnotes DB not found at {GOODNOTES_DB}")
        return []

    # Copy DB to temp file to avoid locking issues
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    shutil.copy2(GOODNOTES_DB, tmp.name)

    # Also copy WAL/SHM if they exist
    for ext in ["-wal", "-shm"]:
        src = GOODNOTES_DB + ext
        if os.path.exists(src):
            shutil.copy2(src, tmp.name + ext)

    try:
        conn = sqlite3.connect(tmp.name)
        conn.row_factory = sqlite3.Row

        # Get all non-deleted documents with page counts
        rows = conn.execute("""
            SELECT
                d.id,
                d.name,
                d.updated_at,
                d.created_at,
                d.document_type,
                COUNT(p.id) as page_count
            FROM documents d
            LEFT JOIN pages p ON p.document_id = d.id AND p.deleted = 0
            WHERE d.deleted = 0 AND d.document_type = 0
            GROUP BY d.id
            HAVING page_count > 1
            ORDER BY d.updated_at DESC
        """).fetchall()

        # Get folder membership to filter out templates
        folder_items = conn.execute("""
            SELECT item_id, root_folder_id, parent_folder_id, item_name
            FROM folder_to_folder_items
            WHERE deleted = 0 AND item_type = 1
        """).fetchall()

        conn.close()
    finally:
        os.unlink(tmp.name)
        for ext in ["-wal", "-shm"]:
            p = tmp.name + ext
            if os.path.exists(p):
                os.unlink(p)

    # Build lookup: doc_id -> root_folder_id
    doc_folders = {}
    for fi in folder_items:
        doc_folders[fi["item_id"]] = fi["root_folder_id"]

    notebooks = []
    for row in rows:
        doc_id = row["id"]
        name = row["name"]

        # Skip templates
        if name in TEMPLATE_NAMES:
            continue
        root = doc_folders.get(doc_id, "")
        if root in TEMPLATE_ROOT_FOLDERS:
            continue

        # Convert timestamp (ms since epoch) to ISO date
        ts = row["updated_at"]
        if ts and ts > 0:
            updated = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000))
        else:
            updated = None

        notebooks.append({
            "id": doc_id,
            "name": name,
            "pages": row["page_count"],
            "updated": updated,
            "has_pdf": False,  # Will be set below
        })

    return notebooks


def find_existing_pdfs(notebooks):
    """Check which notebooks have PDF exports in the journals directory."""
    if not os.path.isdir(JOURNALS_DIR):
        return

    pdfs = set()
    for f in os.listdir(JOURNALS_DIR):
        if f.lower().endswith(".pdf") and not f.startswith("."):
            pdfs.add(f.lower().replace(".pdf", ""))

    for nb in notebooks:
        name_lower = nb["name"].lower()
        # Check exact match or common variations
        if name_lower in pdfs or f"{name_lower}-pdf" in pdfs:
            nb["has_pdf"] = True


def sync_autobackup_pdfs():
    """
    Copy Goodnotes Auto-Backup PDFs from iCloud Drive to the NAS journals dir.
    Uses Finder AppleScript to bypass TCC restrictions on iCloud Drive access.
    """
    # Common Auto-Backup folder names
    backup_folders = [
        "GoodNotes Auto Backup",
        "GoodNotes 5 Auto Backup",
        "Goodnotes Auto Backup",
        "GoodNotes",
    ]

    for folder_name in backup_folders:
        icloud_path = os.path.join(ICLOUD_DRIVE, folder_name)

        # Use Finder to check if folder exists and list PDFs
        script = f'''
        tell application "Finder"
            try
                set backupFolder to folder (POSIX file "{icloud_path}" as alias)
                set pdfFiles to every file of backupFolder whose name extension is "pdf"
                set pdfList to {{}}
                repeat with f in pdfFiles
                    set end of pdfList to (POSIX path of (f as alias))
                end repeat
                return pdfList as string
            on error
                return ""
            end try
        end tell
        '''

        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                pdf_paths = [p.strip() for p in result.stdout.strip().split(", ") if p.strip()]
                copied = 0
                for pdf_path in pdf_paths:
                    if not os.path.isfile(pdf_path):
                        continue
                    dest = os.path.join(JOURNALS_DIR, os.path.basename(pdf_path))
                    # Copy if newer or doesn't exist
                    if not os.path.exists(dest) or os.path.getmtime(pdf_path) > os.path.getmtime(dest):
                        # Use Finder to copy (preserves iCloud access)
                        copy_script = f'''
                        tell application "Finder"
                            try
                                set srcFile to POSIX file "{pdf_path}" as alias
                                set destFolder to POSIX file "{JOURNALS_DIR}" as alias
                                duplicate srcFile to folder destFolder with replacing
                                return "ok"
                            on error errMsg
                                return "error: " & errMsg
                            end try
                        end tell
                        '''
                        cp = subprocess.run(
                            ["osascript", "-e", copy_script],
                            capture_output=True, text=True, timeout=120
                        )
                        if "ok" in cp.stdout:
                            copied += 1
                            print(f"  Copied: {os.path.basename(pdf_path)}")

                if copied > 0:
                    print(f"Synced {copied} PDF(s) from {folder_name}")
                return True
        except Exception as e:
            print(f"  Error checking {folder_name}: {e}")

    return False


def save_metadata(notebooks):
    """Save notebook metadata to JSON for the web app."""
    os.makedirs(JOURNALS_DIR, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump({
            "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "icloud_id": "_3bd21be7957b7a056dd7ca10a999e07e",
            "notebooks": notebooks,
        }, f, indent=2)
    print(f"Saved metadata for {len(notebooks)} notebooks to {META_FILE}")


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Goodnotes sync starting...")

    # 1. Read Goodnotes database
    notebooks = read_goodnotes_db()
    if not notebooks:
        print("No notebooks found")
        return

    print(f"Found {len(notebooks)} notebooks in Goodnotes")

    # 2. Check existing PDFs
    find_existing_pdfs(notebooks)
    with_pdf = sum(1 for nb in notebooks if nb["has_pdf"])
    print(f"  {with_pdf} have PDF exports, {len(notebooks) - with_pdf} need export")

    # 3. Try to sync Auto-Backup PDFs from iCloud Drive
    if sys.platform == "darwin":
        found = sync_autobackup_pdfs()
        if found:
            # Re-check after sync
            find_existing_pdfs(notebooks)

    # 4. Save metadata
    save_metadata(notebooks)

    print(f"[{time.strftime('%H:%M:%S')}] Sync complete")


if __name__ == "__main__":
    main()
