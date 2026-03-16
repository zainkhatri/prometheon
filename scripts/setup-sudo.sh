#!/bin/bash
set -e
BASE=/srv/mergerfs/PROMETHEUS/PROMETHEON

echo "→ Installing sudoers rule..."
cp "$BASE/prometheon-sudoers" /etc/sudoers.d/prometheon
chmod 440 /etc/sudoers.d/prometheon

echo "→ Installing watcher service..."
cp "$BASE/prometheon-watcher.service" /etc/systemd/system/prometheon-watcher.service
systemctl daemon-reload
systemctl enable --now prometheon-watcher

echo "→ Restarting prometheon..."
systemctl restart prometheon

echo "✓ Done. Both services running:"
systemctl is-active prometheon prometheon-watcher
