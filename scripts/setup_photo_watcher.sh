#!/bin/bash
# PROMETHEON Photo Auto-Scanner
# Installs a cron job to pick up new photos every 15 minutes (incremental, fast).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(which python3)"
CRON_CMD="*/15 * * * * $PYTHON $SCRIPT_DIR/photo_scanner.py --incremental >> /var/log/prometheon-scanner.log 2>&1"
CRON_TAG="# PROMETHEON photo auto-scan"

echo "PROMETHEON Photo Auto-Scanner Setup"
echo "====================================="
echo ""
echo "Installs a cron job that runs every 15 minutes."
echo "Only new/changed files are processed — existing photos are untouched."
echo ""
echo "Cron entry:"
echo "  $CRON_CMD"
echo ""

if crontab -l 2>/dev/null | grep -q "PROMETHEON photo auto-scan"; then
    echo "Cron job already installed. Updating..."
    crontab -l 2>/dev/null \
        | grep -v "PROMETHEON photo auto-scan" \
        | grep -v "photo_scanner.py --incremental" \
        | { cat; echo "$CRON_CMD $CRON_TAG"; } \
        | crontab -
else
    echo "Installing cron job..."
    (crontab -l 2>/dev/null; echo "$CRON_CMD $CRON_TAG") | crontab -
fi

echo "Done. Verify with: crontab -l"
echo ""
echo "To pick up photos RIGHT NOW, run:"
echo "  python3 $SCRIPT_DIR/photo_scanner.py --incremental"
