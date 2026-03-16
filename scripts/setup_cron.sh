#!/bin/bash
# PROMETHEON Recycling Bin - Cron Job Installer
# Installs a daily cron job to purge trash items older than 30 days

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(which python3)"
CRON_CMD="0 3 * * * $PYTHON $SCRIPT_DIR/recycling_bin.py purge >> /var/log/prometheon-purge.log 2>&1"
CRON_TAG="# PROMETHEON trash purge"

echo "PROMETHEON Recycling Bin - Cron Setup"
echo "======================================"
echo ""
echo "This will install a daily cron job (runs at 3:00 AM) to purge"
echo "recycling bin items older than 30 days."
echo ""
echo "Cron entry:"
echo "  $CRON_CMD"
echo ""

# Check if already installed
if crontab -l 2>/dev/null | grep -q "PROMETHEON trash purge"; then
    echo "Cron job already installed. Updating..."
    crontab -l 2>/dev/null | grep -v "PROMETHEON trash purge" | grep -v "recycling_bin.py purge" | { cat; echo "$CRON_CMD $CRON_TAG"; } | crontab -
else
    echo "Installing cron job..."
    (crontab -l 2>/dev/null; echo "$CRON_CMD $CRON_TAG") | crontab -
fi

echo "Done. Verify with: crontab -l"
