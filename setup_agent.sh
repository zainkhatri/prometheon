#!/bin/bash
# PROMETHEUS Agent Stack Setup
# Run once on the NAS as root (or with sudo).
# Sets up: /srv/agent-work, /srv/qdrant-data, RAG indexer venv, cron job.
set -e

PROMETHEON_DIR="/srv/mergerfs/PROMETHEUS/PROMETHEON"
RAG_SCRIPT="$PROMETHEON_DIR/rag_indexer.py"
VENV_DIR="/srv/rag-venv"
CRON_USER="root"                   # or whichever user runs the indexer

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   PROMETHEUS Agent Stack Setup       ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ─── 1. Create required directories ──────────────────────────────────────────
echo "→ Creating directories..."
mkdir -p /srv/agent-work
mkdir -p /srv/qdrant-data
mkdir -p /srv/openhands-state
echo "  /srv/agent-work       (writable workspace for OpenHands)"
echo "  /srv/qdrant-data      (Qdrant vector storage)"
echo "  /srv/openhands-state  (OpenHands session state)"

# ─── 2. Pull bge-m3 embedding model ──────────────────────────────────────────
echo ""
echo "→ Pulling bge-m3 embedding model via Ollama..."
ollama pull bge-m3

# ─── 3. Create Python venv for RAG indexer ───────────────────────────────────
echo ""
echo "→ Creating RAG indexer venv at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$PROMETHEON_DIR/rag_requirements.txt"
echo "  Installed: requests, qdrant-client"

# ─── 4. Start Qdrant + OpenHands via docker compose ──────────────────────────
echo ""
echo "→ Starting Qdrant and OpenHands..."
cd "$PROMETHEON_DIR"
docker compose up -d
echo "  Qdrant:    http://localhost:6333"
echo "  OpenHands: http://$(hostname):3000"

# ─── 5. Run initial index ─────────────────────────────────────────────────────
echo ""
echo "→ Running initial RAG index (this will take a while on 3TB)..."
echo "  You can Ctrl-C and let the cron job finish it incrementally."
"$VENV_DIR/bin/python3" "$RAG_SCRIPT" || true

# ─── 6. Install cron job ─────────────────────────────────────────────────────
echo ""
echo "→ Installing cron job (every 6 hours)..."
CRON_CMD="0 */6 * * * $VENV_DIR/bin/python3 $RAG_SCRIPT >> /var/log/rag_indexer.log 2>&1"
# Add only if not already present
( crontab -u "$CRON_USER" -l 2>/dev/null | grep -v "rag_indexer"; echo "$CRON_CMD" ) \
  | crontab -u "$CRON_USER" -
echo "  Cron: $CRON_CMD"
echo "  Logs: /var/log/rag_indexer.log"

# ─── 7. docker compose systemd service ───────────────────────────────────────
echo ""
echo "→ Creating systemd service for docker compose auto-start..."
cat > /etc/systemd/system/prometheus-agent.service << 'EOF'
[Unit]
Description=PROMETHEUS Agent Stack (Qdrant + OpenHands)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/srv/mergerfs/PROMETHEUS/PROMETHEON
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable prometheus-agent.service
echo "  Enabled prometheus-agent.service (starts on boot)"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   Setup complete.                    ║"
echo "║                                      ║"
echo "║   OpenHands → http://$(hostname):3000"
echo "║   Qdrant    → http://localhost:6333  ║"
echo "║   RAG logs  → /var/log/rag_indexer.log"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Remember to update PROMETHEON_DIR in this script"
echo "if your path differs from /opt/prometheon"
