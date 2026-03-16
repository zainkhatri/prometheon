#!/usr/bin/env python3
"""
Auto-export Goodnotes notebooks as PDFs via UI automation.

Reads the Goodnotes projection database to find notebooks needing PDF export,
then uses AppleScript to drive Goodnotes Mac app: search for notebook -> open -> export.

Exported PDFs land in Goodnotes' temp dir, then get copied to NAS journals.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import sys

GOODNOTES_DB = os.path.expanduser(
    "~/Library/Containers/com.goodnotesapp.x/Data/Library/Databases/projection.sqlite"
)
GN_EXPORTS = os.path.expanduser(
    "~/Library/Containers/com.goodnotesapp.x/Data/tmp/Exports"
)
JOURNALS_DIR = "/Volumes/PROMETHEUS/PERSONAL/journals"

TEMPLATE_NAMES = {
    "Text Stamps", "Back To School", "Sticky Notes", "Everyday Stickers",
    "Mind Map Shapes", "Ruled Wide", "Squared Paper", "Ruled Narrow",
    "Bright", "Calligraphr-Template",
}
TEMPLATE_ROOTS = {
    "F6327919-7604-421F-9B60-C38A787F9F42",
    "14AC2082-C07A-4C4F-AF22-A23ACC3B8A5F",
}


def get_notebooks_needing_export():
    if not os.path.exists(GOODNOTES_DB):
        return []
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    shutil.copy2(GOODNOTES_DB, tmp.name)
    for ext in ["-wal", "-shm"]:
        src = GOODNOTES_DB + ext
        if os.path.exists(src):
            shutil.copy2(src, tmp.name + ext)
    try:
        conn = sqlite3.connect(tmp.name)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT d.id, d.name, d.updated_at, COUNT(p.id) as page_count
            FROM documents d
            LEFT JOIN pages p ON p.document_id = d.id AND p.deleted = 0
            WHERE d.deleted = 0 AND d.document_type = 0
            GROUP BY d.id HAVING page_count > 1
            ORDER BY d.updated_at DESC
        """).fetchall()
        folder_items = {
            r["item_id"]: r["root_folder_id"]
            for r in conn.execute(
                "SELECT item_id, root_folder_id FROM folder_to_folder_items WHERE deleted = 0 AND item_type = 1"
            ).fetchall()
        }
        conn.close()
    finally:
        os.unlink(tmp.name)
        for ext in ["-wal", "-shm"]:
            p = tmp.name + ext
            if os.path.exists(p):
                os.unlink(p)

    pdf_mtimes = {}
    if os.path.isdir(JOURNALS_DIR):
        for f in os.listdir(JOURNALS_DIR):
            if f.lower().endswith(".pdf") and not f.startswith("."):
                key = f.lower().replace(".pdf", "").replace("-pdf", "")
                pdf_mtimes[key] = os.path.getmtime(os.path.join(JOURNALS_DIR, f))

    need_export = []
    for row in rows:
        name = row["name"]
        if name in TEMPLATE_NAMES:
            continue
        if folder_items.get(row["id"], "") in TEMPLATE_ROOTS:
            continue
        gn_updated = row["updated_at"] / 1000
        pdf_mtime = pdf_mtimes.get(name.lower(), 0)
        if pdf_mtime == 0 or gn_updated > pdf_mtime:
            need_export.append({
                "id": row["id"], "name": name,
                "pages": row["page_count"], "stale": pdf_mtime > 0,
            })
    return need_export


def _run_applescript(script, timeout=300):
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def _clear_exports():
    if os.path.isdir(GN_EXPORTS):
        for d in os.listdir(GN_EXPORTS):
            p = os.path.join(GN_EXPORTS, d)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)


def _collect_exported_pdf(name):
    """Find and copy exported PDF from Goodnotes temp dir to journals."""
    if not os.path.isdir(GN_EXPORTS):
        return False
    for dirpath, _dirs, files in os.walk(GN_EXPORTS):
        for f in files:
            if f.lower().endswith(".pdf"):
                src = os.path.join(dirpath, f)
                size_mb = os.path.getsize(src) / 1048576
                dest = os.path.join(JOURNALS_DIR, f)
                print(f"    Found: {f} ({size_mb:.0f}MB)")
                shutil.copy2(src, dest)
                print(f"    -> {dest}")
                return True
    return False


def export_notebook(name):
    """Export a specific notebook from Goodnotes via UI automation.

    Flow: activate Goodnotes -> Go > Search -> type name -> Enter to open ->
    File > Export -> select PDF -> click Export -> wait -> cancel save dialog ->
    grab PDF from temp exports dir.
    """
    print(f"  Exporting '{name}'...")
    _clear_exports()

    # Step 1: Open the notebook via search
    open_script = f'''
    tell application "Goodnotes" to activate
    delay 1
    tell application "System Events"
        tell process "Goodnotes"
            -- Go to Library first
            click menu item "Library               " of menu 1 of menu bar item "Go" of menu bar 1
            delay 0.5

            -- Go > Search
            click menu item "Search" of menu 1 of menu bar item "Go" of menu bar 1
            delay 1

            -- Type notebook name
            keystroke "{name}"
            delay 2

            -- Press Return to open first result
            keystroke return
            delay 3

            return name of every window
        end tell
    end tell
    '''
    windows = _run_applescript(open_script, timeout=30)
    print(f"    Windows: {windows}")

    # Step 2: Trigger export
    export_script = '''
    tell application "System Events"
        tell process "Goodnotes"
            -- File > Export
            click menu item "Export..." of menu 1 of menu bar item "File" of menu bar 1
            delay 2

            -- Wait for export sheet
            set s to sheet 1 of window 1
            set allElems to entire contents of s

            -- Click PDF format button (first of the 3 format buttons)
            repeat with elem in allElems
                try
                    if class of elem is button then
                        set p to position of elem
                        if (item 2 of p) > 225 and (item 2 of p) < 300 and (item 1 of p) < 950 then
                            click elem
                            delay 0.3
                            exit repeat
                        end if
                    end if
                end try
            end repeat

            -- Click Export button
            repeat with elem in allElems
                try
                    if class of elem is button and description of elem is "Export" then
                        click elem
                        exit repeat
                    end if
                end try
            end repeat

            return "EXPORT_STARTED"
        end tell
    end tell
    '''
    status = _run_applescript(export_script, timeout=30)
    print(f"    Export: {status}")

    if "EXPORT" not in status:
        print("    Failed to start export")
        return False

    # Step 3: Wait for export to complete (polling)
    print("    Waiting for export...", end="", flush=True)
    for i in range(90):  # up to 3 minutes
        time.sleep(2)
        print(".", end="", flush=True)

        check_script = '''
        tell application "System Events"
            tell process "Goodnotes"
                try
                    set s to sheet 1 of window 1
                    set allElems to entire contents of s
                    repeat with elem in allElems
                        try
                            set d to description of elem
                            if d contains "Save As" then return "SAVE_DIALOG"
                            if d contains "pop up button" then return "SAVE_DIALOG"
                        end try
                        try
                            if class of elem is pop up button then return "SAVE_DIALOG"
                        end try
                    end repeat
                    -- Check if still exporting
                    repeat with elem in allElems
                        try
                            if (description of elem) contains "Exporting" then return "EXPORTING"
                        end try
                    end repeat
                    return "OTHER"
                on error
                    return "NO_SHEET"
                end try
            end tell
        end tell
        '''
        state = _run_applescript(check_script, timeout=10)
        if state == "SAVE_DIALOG":
            print(" done!")
            # Cancel save dialog (we grab from temp)
            _run_applescript('''
            tell application "System Events"
                tell process "Goodnotes"
                    keystroke "." using command down
                end tell
            end tell
            ''', timeout=5)
            time.sleep(2)
            return _collect_exported_pdf(name)
        elif state == "NO_SHEET":
            print(" done (no sheet)!")
            time.sleep(2)
            return _collect_exported_pdf(name)
        elif state != "EXPORTING":
            # Might be save dialog in a different form
            time.sleep(1)

    print(" timeout!")
    # Try to collect anyway
    return _collect_exported_pdf(name)


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Goodnotes PDF export")

    if len(sys.argv) > 1 and sys.argv[1] != "--export":
        # Export specific notebook by name
        name = " ".join(sys.argv[1:])
        os.makedirs(JOURNALS_DIR, exist_ok=True)
        export_notebook(name)
        return

    notebooks = get_notebooks_needing_export()
    if not notebooks:
        print("All notebooks are up to date.")
        return

    print(f"Found {len(notebooks)} notebook(s) needing export:")
    for nb in notebooks:
        status = "stale" if nb["stale"] else "missing"
        print(f"  {nb['name']} ({nb['pages']}p) [{status}]")

    if "--export" not in sys.argv:
        print("\nRun with --export to export all, or pass a notebook name.")
        return

    os.makedirs(JOURNALS_DIR, exist_ok=True)
    for nb in notebooks:
        success = export_notebook(nb["name"])
        if success:
            print(f"    OK!")
        else:
            print(f"    FAILED")
        time.sleep(2)

    print(f"\n[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
