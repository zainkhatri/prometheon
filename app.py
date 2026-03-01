"""Flask server for PROMETHEON NAS Terminal AI Interface."""

import json
import os
import secrets
import functools
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session, redirect, url_for, send_file, abort, make_response
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

from system_info import get_system_info
import llm_interface
import claude_interface
from llm_interface import get_usage_stats
from recycling_bin import trash_file, list_trash, restore as restore_trash

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", secrets.token_hex(32))
app.config['TEMPLATES_AUTO_RELOAD'] = True

LOGIN_PASSWORD = os.getenv("PROMETHEON_PASSWORD", "prometheus")
LOGIN_USER = os.getenv("PROMETHEON_USER", "zainkhatri")
API_TOKEN = os.getenv("PROMETHEON_API_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 86400 * 7

# Store conversation history per session (simple in-memory store)
conversations = {}

# Display-only history per session (role + text, no binary data)
session_display = {}

# ─── Thumbnail cache (RAM + lazy disk generation) ───
# Thumbnails are generated on first request from the original photo and cached
# to disk. Subsequent requests are served from RAM. No pre-generation needed.

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_THUMB_DIR = os.path.join(_APP_DIR, "static", "thumbs")
_THUMB_HQ_DIR = os.path.join(_APP_DIR, "static", "thumbs_hq")
SESSIONS_DIR = os.path.join(_APP_DIR, ".sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(_THUMB_DIR, exist_ok=True)
os.makedirs(_THUMB_HQ_DIR, exist_ok=True)
_thumb_cache = {}
_thumb_cache_lock = threading.Lock()

_landscape_thumbs = set()   # thumb URLs known to be landscape (w > h)
_landscape_index_ready = False

# hash -> original file path (built from index, used for on-demand generation)
_hash_to_path = {}
_hash_index_lock = threading.Lock()


def _build_hash_index(items):
    """Rebuild hash->path reverse map from photo index."""
    mapping = {}
    for item in items:
        orig = item.get("path", "")
        if not orig:
            continue
        for url_key in ("thumb", "thumb_hq"):
            url = item.get(url_key, "")
            if url:
                name = url.rsplit("/", 1)[-1]
                mapping[name] = orig
    with _hash_index_lock:
        _hash_to_path.clear()
        _hash_to_path.update(mapping)


def _jpeg_dimensions(data):
    """Parse width/height from JPEG bytes by reading SOF marker. Returns (w, h) or None."""
    import struct
    if not data or len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    pos = 2
    while pos < len(data) - 1:
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]
        if marker == 0xC0 or marker == 0xC2:
            if pos + 9 < len(data):
                h = struct.unpack(">H", data[pos+5:pos+7])[0]
                w = struct.unpack(">H", data[pos+7:pos+9])[0]
                return (w, h)
            break
        if marker == 0xD9:
            break
        if pos + 3 < len(data):
            length = struct.unpack(">H", data[pos+2:pos+4])[0]
            pos += 2 + length
        else:
            break
    return None


def _build_landscape_index():
    """Scan all cached thumbnails and record which are landscape orientation."""
    global _landscape_index_ready
    count = 0
    with _thumb_cache_lock:
        keys = list(_thumb_cache.keys())
    for key in keys:
        data = _thumb_cache.get(key)
        if not data:
            continue
        dims = _jpeg_dimensions(data)
        if dims and dims[0] > dims[1]:
            name = key[1]  # key is (directory, filename)
            _landscape_thumbs.add(name)
            count += 1
    _landscape_index_ready = True
    print(f"[warmer] Landscape index built: {count} landscape out of {len(keys)} thumbs.")


def _read_thumb(directory, name):
    """Return thumbnail bytes from RAM cache, reading from disk on first access."""
    key = (directory, name)
    data = _thumb_cache.get(key)
    if data is not None:
        return data
    path = os.path.join(directory, name)
    try:
        with open(path, "rb") as f:
            data = f.read()
        with _thumb_cache_lock:
            _thumb_cache[key] = data
        return data
    except OSError:
        return None


def _preload_thumb_batch(items):
    """No-op: thumbnails are now lazy. Kept for API compatibility."""
    pass


@app.before_request
def _serve_thumb_on_demand():
    """Intercept thumbnail requests. Serve from RAM, disk, or generate on demand."""
    path = request.path
    if path.startswith("/static/thumbs_hq/"):
        directory = _THUMB_HQ_DIR
        is_hq = True
    elif path.startswith("/static/thumbs/"):
        directory = _THUMB_DIR
        is_hq = False
    else:
        return None

    if not session.get("authenticated") and not check_bearer_token():
        abort(401)

    name = secure_filename(path.rsplit("/", 1)[-1])

    # 1. RAM cache hit
    data = _read_thumb(directory, name)
    if data is not None:
        return Response(data, mimetype="image/jpeg", headers={
            "Cache-Control": "public, max-age=604800, immutable",
        })

    # 2. Generate from original photo and cache to disk
    orig_path = _hash_to_path.get(name)
    if orig_path and os.path.isfile(orig_path):
        ext = os.path.splitext(orig_path)[1].lower()
        is_video = ext in VIDEO_EXTS
        size = 800 if is_hq else 475
        quality = 2 if is_hq else 3
        out_path = os.path.join(directory, name)
        if gen_thumb(orig_path, out_path, size, quality, is_video):
            data = _read_thumb(directory, name)

    if data is None:
        abort(404)

    return Response(data, mimetype="image/jpeg", headers={
        "Cache-Control": "public, max-age=604800, immutable",
    })


# ─── Session persistence ───

def _strip_images_from_history(history):
    """Replace base64 image blocks with a text placeholder to keep sessions small."""
    clean = []
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    new_content.append({"type": "text", "text": "[image]"})
                else:
                    new_content.append(block)
            clean.append({**msg, "content": new_content})
        else:
            clean.append(msg)
    return clean


def _save_session(session_id, history, model, display):
    """Write session to .sessions/{id}.json."""
    title = "[session]"
    for item in display:
        if item["role"] == "user" and item["text"] != "[image]":
            title = item["text"][:60]
            break
    data = {
        "id": session_id,
        "title": title,
        "updated_at": time.time(),
        "updated_at_str": datetime.now().strftime("%b %d %H:%M"),
        "model": model,
        "history": _strip_images_from_history(history),
        "display": display,
    }
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except OSError:
        pass


def _load_session(session_id):
    """Load session JSON from disk. Returns dict or None."""
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _list_sessions(limit=20):
    """Return list of session summaries sorted by updated_at desc."""
    sessions = []
    try:
        filenames = os.listdir(SESSIONS_DIR)
    except OSError:
        return []
    for fname in filenames:
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            sessions.append({
                "id": data["id"],
                "title": data.get("title", "[untitled]"),
                "updated_at": data.get("updated_at", 0),
                "updated_at_str": data.get("updated_at_str", ""),
                "model": data.get("model", "claude"),
                "message_count": len(data.get("display", [])),
            })
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions[:limit]


def check_bearer_token():
    """Check if request has a valid bearer token."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and API_TOKEN:
        return secrets.compare_digest(auth_header[7:], API_TOKEN)
    return False


def require_auth(f):
    """Decorator to require session login or bearer token."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if session.get("authenticated") or check_bearer_token():
            return f(*args, **kwargs)
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"error": "Not authenticated"}), 401
        return redirect(url_for("login_page"))
    return decorated


@app.route("/login", methods=["GET"])
def login_page():
    if session.get("authenticated"):
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username", "")
    password = data.get("password", "")

    if password == LOGIN_PASSWORD:
        session["authenticated"] = True
        session["username"] = "zain"
        session.permanent = False
        return jsonify({"success": True, "username": "zain"})
    else:
        return jsonify({"success": False, "error": "Authentication failed."}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/")
@require_auth
def home():
    return render_template("home.html")


@app.route("/terminal")
@require_auth
def terminal():
    return render_template("index.html", username=session.get("username", LOGIN_USER))


@app.route("/api/system-info")
@require_auth
def system_info():
    info = get_system_info()
    info["api_usage"] = get_usage_stats()
    return jsonify(info)


@app.route("/api/chat", methods=["POST"])
@require_auth
def chat_endpoint():
    data = request.json
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    image_b64 = data.get("image", "")
    image_mime = data.get("image_mime", "image/jpeg")

    if not message and not image_b64:
        return jsonify({"error": "Empty message"}), 400

    if session_id not in conversations:
        conversations[session_id] = []

    history = conversations[session_id]

    def generate():
        display = session_display.setdefault(session_id, [])
        display.append({"role": "user", "text": message or "[image]"})
        full_text = ""

        if not ANTHROPIC_API_KEY:
            stream = [
                {"type": "text", "content": "❌ Claude API key not configured. Add ANTHROPIC_API_KEY to .env"},
                {"type": "done"},
            ]
        else:
            stream = claude_interface.chat_stream(
                message, history, ANTHROPIC_API_KEY,
                image_b64 or None, image_mime
            )

        for event in stream:
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] == "text":
                full_text += event["content"]
            elif event["type"] == "done":
                if full_text:
                    display.append({"role": "assistant", "text": full_text})
                _save_session(session_id, history, "claude", display)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/sessions")
@require_auth
def api_sessions_list():
    return jsonify(_list_sessions())


@app.route("/api/sessions/<session_id>")
@require_auth
def api_session_load(session_id):
    data = _load_session(session_id)
    if data is None:
        return jsonify({"error": "Session not found"}), 404
    # Restore into memory so subsequent messages use this history
    conversations[session_id] = data.get("history", [])
    session_display[session_id] = data.get("display", [])
    return jsonify(data)


TRASH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".prometheon-trash")


@app.route("/api/trash")
@require_auth
def trash_list():
    return jsonify(list_trash())


@app.route("/api/trash/preview/<trash_name>")
@require_auth
def trash_preview(trash_name):
    """Serve a trashed file for thumbnail preview."""
    safe_name = secure_filename(trash_name)
    filepath = os.path.join(TRASH_DIR, safe_name)
    if not os.path.isfile(filepath):
        abort(404)
    return send_file(filepath, conditional=True)


@app.route("/api/trash/restore", methods=["POST"])
@require_auth
def trash_restore():
    data = request.json
    trash_name = data.get("trash_name", "")
    if not trash_name:
        return jsonify({"error": "No trash_name provided"}), 400
    result = restore_trash(trash_name)
    return jsonify(result)


@app.route("/api/photos/trash", methods=["POST"])
@require_auth
def trash_photo():
    """Trash a single photo by its hash (thumbnail filename without .jpg)."""
    data = request.json or {}
    h = data.get("hash", "")
    if not h:
        return jsonify({"error": "No hash provided"}), 400
    name = h + ".jpg" if not h.endswith(".jpg") else h
    orig_path = _hash_to_path.get(name)
    if not orig_path:
        return jsonify({"error": "Photo not found"}), 404
    result = trash_file(orig_path)
    if result.get("success"):
        # Invalidate caches so the photo disappears
        _photo_cache["data"] = None
        _month_cache["data"] = None
        _month_json_cache["data"] = None
    return jsonify(result)


# ─── Photo Gallery ───

PHOTO_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo_index.json")
_photo_cache = {"data": None, "mtime": 0}
_summary_cache = {"data": None, "mtime": 0}
_month_cache = {"data": None, "mtime": 0}
_month_json_cache = {"data": None, "mtime": 0}


def load_photo_index():
    """Load photo index with simple file-mtime cache."""
    try:
        mtime = os.path.getmtime(PHOTO_INDEX_PATH)
    except OSError:
        return []
    if _photo_cache["data"] is not None and _photo_cache["mtime"] == mtime:
        return _photo_cache["data"]
    with open(PHOTO_INDEX_PATH) as f:
        _photo_cache["data"] = json.load(f)
    _photo_cache["mtime"] = mtime
    # Rebuild reverse hash map for on-demand thumb generation
    threading.Thread(target=_build_hash_index, args=(_photo_cache["data"],), daemon=True).start()
    _month_cache["data"] = None
    _month_json_cache["data"] = None
    return _photo_cache["data"]


def load_month_index():
    """Return dict of month_key -> [items], cached alongside photo index."""
    try:
        mtime = os.path.getmtime(PHOTO_INDEX_PATH)
    except OSError:
        return {}
    if _month_cache["data"] is not None and _month_cache["mtime"] == mtime:
        return _month_cache["data"]
    items = load_photo_index()
    by_month = {}
    for item in sorted(items, key=lambda x: x.get("date", 0), reverse=True):
        try:
            dt = datetime.fromtimestamp(item.get("date", 0))
            key = dt.strftime("%Y-%m")
        except Exception:
            continue
        if key not in by_month:
            by_month[key] = []
        by_month[key].append(item)
    _month_cache["data"] = by_month
    _month_cache["mtime"] = mtime
    _month_json_cache["data"] = None  # invalidate serialized cache
    return by_month


def _get_hidden_hashes():
    """Return set of photo hashes belonging to hidden face clusters."""
    clusters = _ai.get("face_clusters")
    if not clusters:
        return set()
    hidden = set()
    for c in clusters.values():
        if c.get("hidden"):
            hidden.update(c.get("photo_hashes", []))
    return hidden


def _get_screenshot_hashes():
    """Return set of photo hashes classified as screenshots/documents."""
    return _ai.get("screenshot_hashes") or set()


@app.route("/api/photos/thumb-bundle")
@require_auth
def thumb_bundle():
    """Stream ALL thumbnails as one binary blob for the service worker to cache.
    Format per entry: [2B url_len][url bytes][4B data_len][jpeg bytes]
    One request replaces 30k individual thumbnail requests.
    """
    import struct
    items = load_photo_index()
    images = [item for item in items if item.get("thumb") and item.get("type") != "video"]
    total = len(images)

    def generate():
        for item in images:
            thumb_url = item["thumb"]
            name = secure_filename(thumb_url.rsplit("/", 1)[-1])
            data = _read_thumb(_THUMB_DIR, name)
            if data is None:
                continue
            url_bytes = thumb_url.encode("utf-8")
            yield struct.pack(">H", len(url_bytes)) + url_bytes + struct.pack(">I", len(data)) + data

    return Response(
        stream_with_context(generate()),
        mimetype="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thumb-Count": str(total),
        },
    )


@app.route("/sw.js")
def service_worker():
    """Serve SW from root so it has scope over /static/thumbs/*."""
    resp = make_response(send_file(os.path.join(_APP_DIR, "static", "sw.js")))
    resp.headers["Content-Type"] = "application/javascript"
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/photos/warm/<month_key>")
@require_auth
def warm_month(month_key):
    """Trigger background thumb generation for a specific month (YYYY-MM).
    Called by the frontend when a month is clicked so thumbs are ready when images load.
    """
    items = load_photo_index()
    month_items = [i for i in items if
        __import__("datetime").datetime.fromtimestamp(i["date"]).strftime("%Y-%m") == month_key]
    if not month_items:
        return jsonify({"status": "empty"})
    threading.Thread(target=_warm_recent_months, args=(month_items, 99), daemon=True).start()
    return jsonify({"status": "warming", "count": len(month_items)})


@app.route("/photos")
@require_auth
def photos_page():
    hidden = _get_hidden_hashes()
    screenshots = _get_screenshot_hashes()
    exclude = hidden | screenshots
    all_items = sorted(load_photo_index(), key=lambda x: x.get('date', 0), reverse=True)
    items = [i for i in all_items if not (exclude and i.get("thumb", "").rsplit("/", 1)[-1].replace(".jpg", "") in exclude)]
    groups = OrderedDict()
    for item in items:
        try:
            dt = datetime.fromtimestamp(item.get("date", 0))
        except Exception:
            continue
        month_key = dt.strftime("%Y-%m")
        if month_key not in groups:
            groups[month_key] = {
                "month": dt.strftime("%B %Y"),
                "month_key": month_key,
                "count": 0,
                "cover": item.get("thumb", ""),
            }
        groups[month_key]["count"] += 1
    # Inline the first 3 months of items so they render without an API call
    by_month = load_month_index()
    group_keys = list(groups.keys())
    if exclude:
        inline_items = {key: [i for i in by_month.get(key, []) if i.get("thumb", "").rsplit("/", 1)[-1].replace(".jpg", "") not in exclude] for key in group_keys[:3]}
    else:
        inline_items = {key: by_month.get(key, []) for key in group_keys[:3]}
    photo_data_json = json.dumps({
        "groups": list(groups.values()),
        "total": len(items),
        "inline": inline_items,
    })
    first_thumbs = [item["thumb"] for item in items[:30] if item.get("thumb")]
    resp = make_response(render_template("photos.html",
        photo_data_json=photo_data_json,
        first_thumbs=first_thumbs,
    ))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/photos/month/<month_key>")
@require_auth
def api_photos_month(month_key):
    """Return all items for a specific month."""
    by_month = load_month_index()
    items = by_month.get(month_key, [])
    exclude = _get_hidden_hashes() | _get_screenshot_hashes()
    if exclude:
        items = [i for i in items if i.get("thumb", "").rsplit("/", 1)[-1].replace(".jpg", "") not in exclude]
    return jsonify(items)


@app.route("/api/photos/all-months")
@require_auth
def api_photos_all_months():
    """Return every month's items in one shot for bulk client-side caching.

    Response is gzip-compressed (~1-2 MB) and pre-serialized so repeated
    calls are near-instant. The client fetches this once after page load
    to populate its monthCache, making all subsequent month clicks instant.
    """
    import gzip as _gzip

    try:
        mtime = os.path.getmtime(PHOTO_INDEX_PATH)
    except OSError:
        return jsonify({})

    exclude = _get_hidden_hashes() | _get_screenshot_hashes()
    if _month_json_cache["data"] is not None and _month_json_cache["mtime"] == mtime and not exclude:
        raw = _month_json_cache["data"]
    else:
        by_month = load_month_index()
        if exclude:
            by_month = {k: [i for i in v if i.get("thumb", "").rsplit("/", 1)[-1].replace(".jpg", "") not in exclude] for k, v in by_month.items()}
        raw = json.dumps(by_month).encode("utf-8")
        if not exclude:
            _month_json_cache["data"] = raw
            _month_json_cache["mtime"] = mtime

    if "gzip" in request.headers.get("Accept-Encoding", ""):
        compressed = _gzip.compress(raw, compresslevel=4)
        return Response(compressed, mimetype="application/json", headers={
            "Content-Encoding": "gzip",
            "Cache-Control": "public, max-age=120",
        })
    return Response(raw, mimetype="application/json", headers={
        "Cache-Control": "public, max-age=120",
    })


@app.route("/api/photos")
@require_auth
def api_photos():
    page = request.args.get("page", 0, type=int)
    per_page = 200
    items = sorted(load_photo_index(), key=lambda x: x.get('date', 0), reverse=True)
    total = len(items)
    start = page * per_page
    end = start + per_page
    page_items = items[start:end]

    # Preload this page + next page thumbnails into RAM in background
    preload_slice = items[start:end + per_page]
    threading.Thread(target=_preload_thumb_batch, args=(preload_slice,), daemon=True).start()

    # Group by month
    groups = OrderedDict()
    for item in page_items:
        dt = datetime.fromtimestamp(item["date"])
        month_key = dt.strftime("%Y-%m")
        month_label = dt.strftime("%B %Y")
        if month_key not in groups:
            groups[month_key] = {"month": month_label, "month_key": month_key, "items": []}
        groups[month_key]["items"].append(item)

    return jsonify({"groups": list(groups.values()), "total": total, "page": page})


@app.route("/api/photos/all")
@require_auth
def api_photos_all():
    """Return every photo grouped by month in one shot — for LAN use where latency is negligible."""
    items = sorted(load_photo_index(), key=lambda x: x.get('date', 0), reverse=True)
    total = len(items)

    # Preload all thumbnails into RAM in background
    threading.Thread(target=_preload_thumb_batch, args=(items,), daemon=True).start()

    groups = OrderedDict()
    for item in items:
        dt = datetime.fromtimestamp(item["date"])
        month_key = dt.strftime("%Y-%m")
        month_label = dt.strftime("%B %Y")
        if month_key not in groups:
            groups[month_key] = {"month": month_label, "month_key": month_key, "items": []}
        groups[month_key]["items"].append(item)

    return jsonify({"groups": list(groups.values()), "total": total})


_CAMERA_FOLDERS = {"S95", "RX100", "GX9", "FUJI", "PIXPRO"}


def _is_camera_source(path):
    """True if the photo came from a digital camera folder (not phone)."""
    parts = path.upper().split("/")
    return any(p in _CAMERA_FOLDERS for p in parts)


def _is_landscape(thumb_url):
    """Check if a thumbnail is landscape using the pre-built index."""
    if not _landscape_index_ready or not thumb_url:
        return False
    name = thumb_url.rsplit("/", 1)[-1]
    return name in _landscape_thumbs


def _pick_covers(candidates, max_covers=6):
    """Pick cover photos preferring landscape camera shots.

    Priority: landscape camera > landscape phone > any camera > fallback.
    """
    import random

    cam = [c for c in candidates if c["is_camera"]]
    phone = [c for c in candidates if not c["is_camera"]]
    random.shuffle(cam)
    random.shuffle(phone)

    landscape_cam = [c["thumb"] for c in cam if _is_landscape(c["thumb"])]
    landscape_phone = [c["thumb"] for c in phone if _is_landscape(c["thumb"])]

    pool = landscape_cam[:max_covers]
    if len(pool) < max_covers:
        pool.extend(landscape_phone[:max_covers - len(pool)])
    if len(pool) < max_covers:
        cam_thumbs = [c["thumb"] for c in cam if c["thumb"] not in pool]
        pool.extend(cam_thumbs[:max_covers - len(pool)])
    if len(pool) < max_covers:
        remaining = [c["thumb"] for c in candidates if c["thumb"] not in pool]
        random.shuffle(remaining)
        pool.extend(remaining[:max_covers - len(pool)])

    random.shuffle(pool)
    return pool[:max_covers]


@app.route("/api/photos/summary")
@require_auth
def api_photos_summary():
    """Pre-computed month/year summary, cached by index mtime."""
    try:
        mtime = os.path.getmtime(PHOTO_INDEX_PATH)
    except OSError:
        return jsonify({"months": [], "years": [], "total": 0})
    if _summary_cache["data"] is not None and _summary_cache["mtime"] == mtime:
        return jsonify(_summary_cache["data"])

    items = load_photo_index()
    months = OrderedDict()
    month_candidates = {}  # month_key -> list of {thumb, is_camera}
    MAX_COVERS = 6
    for item in items:
        if item.get("type") == "video":
            continue
        dt = datetime.fromtimestamp(item["date"])
        mk = dt.strftime("%Y-%m")
        if mk not in months:
            months[mk] = {
                "month_key": mk,
                "month": dt.strftime("%B %Y"),
                "year": dt.year,
                "count": 0,
                "covers": [],
            }
            month_candidates[mk] = []
        months[mk]["count"] += 1
        thumb = item.get("thumb_hq") or item.get("thumb", "")
        if thumb:
            month_candidates[mk].append({
                "thumb": thumb,
                "is_camera": _is_camera_source(item.get("path", "")),
            })

    # Also count videos toward month totals
    for item in items:
        if item.get("type") == "video":
            dt = datetime.fromtimestamp(item["date"])
            mk = dt.strftime("%Y-%m")
            if mk in months:
                months[mk]["count"] += 1

    # Pick covers using CLIP aesthetic scoring (falls back to landscape heuristic)
    for mk, cands in month_candidates.items():
        months[mk]["covers"] = _pick_aesthetic_covers(cands, MAX_COVERS)

    # Build year summary
    years = OrderedDict()
    for mk, mo in months.items():
        yk = str(mo["year"])
        if yk not in years:
            years[yk] = {"year": mo["year"], "count": 0, "covers": [], "months": 0, "_candidates": []}
        years[yk]["count"] += mo["count"]
        years[yk]["months"] += 1
        years[yk]["_candidates"].extend(month_candidates.get(mk, []))

    for yk, yr in years.items():
        yr["covers"] = _pick_aesthetic_covers(yr["_candidates"], MAX_COVERS)
        del yr["_candidates"]

    # Sort newest first
    sorted_months = sorted(months.values(), key=lambda m: m["month_key"], reverse=True)
    sorted_years = sorted(years.values(), key=lambda y: y["year"], reverse=True)

    photo_count = sum(1 for i in items if i.get("type") != "video")
    video_count = sum(1 for i in items if i.get("type") == "video")
    result = {
        "months": sorted_months,
        "years": sorted_years,
        "total": len(items),
        "photos": photo_count,
        "videos": video_count,
    }
    _summary_cache["data"] = result
    _summary_cache["mtime"] = mtime
    return jsonify(result)


PATH_ALIASES = [
    ("/srv/mergerfs/PROMETHEUS/", "/Volumes/PROMETHEUS/"),
]

MEDIA_CACHE_MAX_AGE = 86400 * 7  # 7 days


def resolve_media_path(filepath):
    """Translate path aliases before hitting the filesystem."""
    for src, dst in PATH_ALIASES:
        if filepath.startswith(src):
            return dst + filepath[len(src):]
        elif filepath.startswith(dst):
            return src + filepath[len(dst):]
    return None


@app.route("/media/<path:filepath>")
@require_auth
def serve_media(filepath):
    import mimetypes
    filepath = "/" + filepath
    # Try alias first to avoid slow stat() on non-existent mount paths
    alt = resolve_media_path(filepath)
    if alt and os.path.isfile(alt):
        filepath = alt
    elif not os.path.isfile(filepath):
        abort(404)
    abs_path = os.path.abspath(filepath)
    allowed = ["/srv/mergerfs/PROMETHEUS/PHOTOS/", "/Volumes/PROMETHEUS/PHOTOS/"]
    if not any(abs_path.startswith(prefix) for prefix in allowed):
        abort(403)
    # X-Accel-Redirect: let nginx serve file directly (zero-copy sendfile)
    # Falls back to Flask send_file when not behind nginx
    if os.environ.get("NGINX_ACCEL"):
        mime = mimetypes.guess_type(abs_path)[0] or "application/octet-stream"
        resp = make_response("")
        resp.headers["X-Accel-Redirect"] = "/internal-media" + abs_path
        resp.headers["Content-Type"] = mime
        resp.headers["Cache-Control"] = f"public, max-age={MEDIA_CACHE_MAX_AGE}"
        return resp
    return send_file(filepath, conditional=True, max_age=MEDIA_CACHE_MAX_AGE)




# ─── Download tokens (one-time URLs so browser handles download natively) ───
_dl_tokens      = {}  # token -> {"path": str, "expires": float}         (single file)
_dl_zip_tokens  = {}  # token -> {"files": [(path, name)], "expires": float}  (zip)
_dl_tokens_lock = threading.Lock()


def _prune_tokens():
    """Remove expired tokens."""
    now = time.time()
    with _dl_tokens_lock:
        for t in [t for t, v in _dl_tokens.items() if v["expires"] < now]:
            _dl_tokens.pop(t)
        for t in [t for t, v in _dl_zip_tokens.items() if v["expires"] < now]:
            _dl_zip_tokens.pop(t)


def _streaming_zip(files):
    """
    Generator that yields raw bytes of a valid ZIP file without any temp file.
    Uses data descriptors (flag bit 3) so CRC/sizes are written after each file,
    allowing true streaming with no seek-back.
    files: list of (real_path, arcname) tuples.
    """
    import struct, zlib as _zlib

    central = []
    pos = 0

    for real_path, arcname in files:
        try:
            stat = os.stat(real_path)
        except OSError:
            continue
        name_b = arcname.encode("utf-8")
        t = time.localtime(stat.st_mtime)
        dos_time = (t.tm_sec >> 1) | (t.tm_min << 5) | (t.tm_hour << 11)
        dos_date = t.tm_mday | (t.tm_mon << 5) | ((t.tm_year - 1980) << 9)

        # Local file header — CRC/sizes are 0; data descriptor follows file data
        lf = struct.pack(
            "<4sHHHHHIIIHH",
            b"PK\x03\x04", 20, 0x0808, 0,
            dos_time, dos_date, 0, 0, 0,
            len(name_b), 0,
        ) + name_b
        yield lf
        file_offset = pos
        pos += len(lf)

        crc, size = 0, 0
        try:
            with open(real_path, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 16)
                    if not chunk:
                        break
                    crc = _zlib.crc32(chunk, crc) & 0xFFFFFFFF
                    size += len(chunk)
                    pos += len(chunk)
                    yield chunk
        except OSError:
            pass

        dd = struct.pack("<4sIII", b"PK\x07\x08", crc, size, size)
        yield dd
        pos += len(dd)

        central.append((name_b, dos_time, dos_date, crc, size, file_offset))

    cd_offset, cd_size = pos, 0
    for name_b, dos_time, dos_date, crc, size, offset in central:
        cde = struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02", 20, 20, 0x0808, 0,
            dos_time, dos_date, crc, size, size,
            len(name_b), 0, 0, 0, 0, 0, offset,
        ) + name_b
        yield cde
        cd_size += len(cde)

    yield struct.pack(
        "<4sHHHHIIH",
        b"PK\x05\x06", 0, 0,
        len(central), len(central),
        cd_size, cd_offset, 0,
    )


def _resolve_photo_path(p):
    for src, dst in PATH_ALIASES:
        if p.startswith(src):
            alt = dst + p[len(src):]
            if os.path.isfile(alt):
                return alt
        elif p.startswith(dst):
            alt = src + p[len(dst):]
            if os.path.isfile(alt):
                return alt
    return p if os.path.isfile(p) else None


_PHOTO_ALLOWED = ["/srv/mergerfs/PROMETHEUS/PHOTOS/", "/Volumes/PROMETHEUS/PHOTOS/"]


@app.route("/api/photos/download-zip")
@require_auth
def download_photos_zip():
    """Stream a zip of selected photos. Paths come as repeated ?p= query params."""
    paths = request.args.getlist("p")
    if not paths:
        abort(400)
    if len(paths) > 500:
        abort(400)
    valid = []
    seen_names = {}
    for raw_path in paths:
        rp = _resolve_photo_path(raw_path)
        if not rp:
            continue
        if not any(os.path.abspath(rp).startswith(pfx) for pfx in _PHOTO_ALLOWED):
            continue
        name = os.path.basename(rp)
        if name in seen_names:
            seen_names[name] += 1
            base, ext = os.path.splitext(name)
            name = f"{base}_{seen_names[name]}{ext}"
        else:
            seen_names[name] = 0
        valid.append((rp, name))
    if not valid:
        abort(400)
    return Response(
        stream_with_context(_streaming_zip(valid)),
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=photos.zip"},
    )


def _to_jpeg_bytes(src_path):
    """Convert any image (including HEIC) to JPEG bytes. Returns None on failure."""
    try:
        from PIL import Image
        import io
        ext = src_path.rsplit(".", 1)[-1].lower()
        if ext in ("heic", "heif"):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
        img = Image.open(src_path)
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return buf
    except Exception:
        return None


@app.route("/api/photos/download")
@require_auth
def download_photo_direct():
    """Download a single photo. ?p=path  Optional ?fmt=jpeg to force JPEG conversion."""
    raw_path = request.args.get("p", "").strip()
    rp = _resolve_photo_path(raw_path)
    if not rp:
        abort(404)
    if not any(os.path.abspath(rp).startswith(pfx) for pfx in _PHOTO_ALLOWED):
        abort(403)
    want_jpeg = request.args.get("fmt") == "jpeg"
    ext = rp.rsplit(".", 1)[-1].lower()
    if want_jpeg and ext not in ("jpg", "jpeg"):
        buf = _to_jpeg_bytes(rp)
        if buf:
            stem = os.path.splitext(os.path.basename(rp))[0]
            return send_file(buf, mimetype="image/jpeg", as_attachment=True,
                             download_name=stem + ".jpg")
    return send_file(rp, as_attachment=True, download_name=os.path.basename(rp))


# ─── Photo Upload (iPhone Sync) ───

from photo_scanner import gen_thumb, get_media_date, hash_path, IMAGE_EXTS, VIDEO_EXTS, ALL_EXTS, THUMB_DIR, THUMB_HQ_DIR, PHOTOS_ROOT


@app.route("/api/upload", methods=["POST"])
@require_auth
def upload_photo():
    """Accept photo/video upload, thumbnail it, and add to the index."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALL_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    # Save to iPhone/<YYYY>/<MM>/
    now = datetime.now()
    dest_dir = os.path.join(PHOTOS_ROOT, "iPhone", str(now.year), f"{now.month:02d}")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    # Skip duplicates
    if os.path.exists(dest_path):
        return jsonify({"status": "skipped", "reason": "duplicate", "path": dest_path}), 200

    f.save(dest_path)

    # Generate thumbnails
    rel_path = os.path.relpath(dest_path, PHOTOS_ROOT)
    thumb_name = hash_path(rel_path) + ".jpg"
    thumb_path = os.path.join(THUMB_DIR, thumb_name)
    thumb_hq_path = os.path.join(THUMB_HQ_DIR, thumb_name)
    is_video = ext in VIDEO_EXTS

    os.makedirs(THUMB_DIR, exist_ok=True)
    os.makedirs(THUMB_HQ_DIR, exist_ok=True)
    gen_thumb(dest_path, thumb_path, 475, 3, is_video)
    gen_thumb(dest_path, thumb_hq_path, 800, 2, is_video)

    # Extract EXIF date
    date = get_media_date(dest_path)
    if date is None:
        date = os.path.getmtime(dest_path)

    media_type = "video" if is_video else "image"
    entry = {
        "path": dest_path,
        "thumb": f"/static/thumbs/{thumb_name}",
        "thumb_hq": f"/static/thumbs_hq/{thumb_name}",
        "date": date,
        "type": media_type,
    }

    # Insert into index in sorted position (newest first = descending by date)
    items = load_photo_index()
    insert_pos = len(items)
    for i, item in enumerate(items):
        if date >= item["date"]:
            insert_pos = i
            break
    items.insert(insert_pos, entry)

    with open(PHOTO_INDEX_PATH, "w") as idx_f:
        json.dump(items, idx_f)

    # Invalidate caches
    _photo_cache["data"] = None
    _photo_cache["mtime"] = 0
    _summary_cache["data"] = None
    _summary_cache["mtime"] = 0
    _month_json_cache["data"] = None

    return jsonify({"status": "ok", "entry": entry}), 201


# ─── Photo Delete (trash) ───

@app.route("/api/photos/rotate", methods=["POST"])
@require_auth
def rotate_photo():
    """Rotate photo thumbnails. Supports single or batch rotation.

    Body: {"hash": "abc", "direction": "cw"} — single photo
      or: {"hashes": ["abc","def"], "direction": "ccw"} — batch
    direction: "cw" (default, 90° clockwise) or "ccw" (90° counter-clockwise)
    """
    from PIL import Image
    data = request.json
    direction = data.get("direction", "cw")
    angle = -90 if direction == "cw" else 90

    hashes = data.get("hashes", [])
    if not hashes:
        h = data.get("hash", "")
        if h:
            hashes = [h]
    if not hashes:
        return jsonify({"error": "No hash provided"}), 400

    results = {"rotated": 0, "errors": []}
    for photo_hash in hashes:
        ok = False
        for tdir in [_THUMB_DIR, _THUMB_HQ_DIR]:
            tp = os.path.join(tdir, photo_hash + ".jpg")
            if os.path.exists(tp):
                try:
                    img = Image.open(tp)
                    img = img.rotate(angle, expand=True)
                    img.save(tp, "JPEG", quality=85, optimize=True)
                    ok = True
                except Exception as e:
                    results["errors"].append(f"{photo_hash}: {e}")
        if ok:
            results["rotated"] += 1
            name = photo_hash + ".jpg"
            with _thumb_cache_lock:
                _thumb_cache.pop((_THUMB_DIR, name), None)
                _thumb_cache.pop((_THUMB_HQ_DIR, name), None)

    return jsonify({"success": True, **results})


@app.route("/api/photos/delete", methods=["POST"])
@require_auth
def delete_photo():
    """Move a photo to the recycling bin and remove from index."""
    data = request.json
    photo_path = data.get("path", "")
    if not photo_path:
        return jsonify({"error": "No path provided"}), 400

    # Security: only allow deleting from known photo directories
    abs_path = os.path.abspath(photo_path)
    allowed = ["/srv/mergerfs/PROMETHEUS/PHOTOS/", "/Volumes/PROMETHEUS/PHOTOS/"]
    if not any(abs_path.startswith(prefix) for prefix in allowed):
        return jsonify({"error": "Not allowed"}), 403

    # Move to recycling bin
    result = trash_file(photo_path)
    if not result["success"]:
        return jsonify({"error": result["error"]}), 400

    # Remove from index
    items = load_photo_index()
    items = [i for i in items if i["path"] != photo_path]
    with open(PHOTO_INDEX_PATH, "w") as f:
        json.dump(items, f)

    # Remove thumbnails
    rel_path = os.path.relpath(photo_path, PHOTOS_ROOT)
    thumb_name = hash_path(rel_path) + ".jpg"
    for tdir in [THUMB_DIR, THUMB_HQ_DIR]:
        tp = os.path.join(tdir, thumb_name)
        if os.path.exists(tp):
            os.unlink(tp)

    # Invalidate caches
    _photo_cache["data"] = None
    _photo_cache["mtime"] = 0
    _summary_cache["data"] = None
    _summary_cache["mtime"] = 0
    _month_json_cache["data"] = None

    return jsonify({"status": "ok", "trash_name": result["trash_name"]})


def _startup_preload():
    """Build hash index, month index, and warm ALL thumbs in background after startup."""
    try:
        items = load_photo_index()
        if not items:
            return
        _build_hash_index(items)
        load_month_index()
        threading.Thread(target=_warm_all_thumbs, args=(items,), daemon=True).start()
    except Exception:
        pass


def _warm_all_thumbs(items):
    """Generate every missing thumbnail and load all into RAM cache.
    Processes newest photos first so recent months are ready fastest.
    """
    # First: load already-existing thumbs into RAM cache
    existing_loaded = 0
    for item in items:
        url = item.get("thumb", "")
        if not url:
            continue
        name = url.rsplit("/", 1)[-1]
        path = os.path.join(_THUMB_DIR, name)
        if os.path.exists(path) and (_THUMB_DIR, name) not in _thumb_cache:
            _read_thumb(_THUMB_DIR, name)
            existing_loaded += 1
    if existing_loaded:
        print(f"[warmer] Loaded {existing_loaded} existing thumbs into RAM.")

    _build_landscape_index()
    _summary_cache["data"] = None  # force covers to use landscape data

    # Then: generate missing thumbs and add them to RAM too
    missing = []
    for item in items:
        orig = item.get("path", "")
        if not orig:
            continue
        url = item.get("thumb", "")
        if not url:
            continue
        name = url.rsplit("/", 1)[-1]
        if not os.path.exists(os.path.join(_THUMB_DIR, name)):
            missing.append(item)

    if not missing:
        print(f"[warmer] All {len(items)} thumbs on disk and in RAM.")
        return

    print(f"[warmer] Generating {len(missing)} missing thumbs...")

    def _gen(item):
        orig = item["path"]
        if not os.path.isfile(orig):
            return
        ext = os.path.splitext(orig)[1].lower()
        is_video = ext in VIDEO_EXTS
        url = item["thumb"]
        name = url.rsplit("/", 1)[-1]
        out = os.path.join(_THUMB_DIR, name)
        if gen_thumb(orig, out, 475, 3, is_video):
            _read_thumb(_THUMB_DIR, name)  # immediately cache in RAM

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(_gen, missing):
            done += 1
            if done % 1000 == 0:
                print(f"[warmer] {done}/{len(missing)} thumbs done")

    print(f"[warmer] Done — all thumbs on disk and in RAM.")
    _build_landscape_index()
    _summary_cache["data"] = None  # force re-generation with landscape data


def _warm_recent_months(items, months=3):
    """Generate missing thumbs for the N most recent months in a thread pool."""
    from datetime import datetime
    from collections import OrderedDict

    # Group by month, take the most recent N
    groups = OrderedDict()
    for item in items:
        mk = datetime.fromtimestamp(item["date"]).strftime("%Y-%m")
        groups.setdefault(mk, []).append(item)

    recent_keys = sorted(groups.keys(), reverse=True)[:months]
    to_warm = [item for k in recent_keys for item in groups[k]]

    missing = []
    for item in to_warm:
        for url_key, tdir in (("thumb", _THUMB_DIR), ("thumb_hq", _THUMB_HQ_DIR)):
            url = item.get(url_key, "")
            if not url:
                continue
            name = url.rsplit("/", 1)[-1]
            if not os.path.exists(os.path.join(tdir, name)):
                missing.append((item, url_key, tdir, name))

    if not missing:
        return

    print(f"[warmer] Pre-generating {len(missing)} thumbs for recent {months} months...")

    def _gen_one(args):
        item, url_key, tdir, name = args
        orig = item.get("path", "")
        if not orig or not os.path.isfile(orig):
            return
        is_hq = url_key == "thumb_hq"
        ext = os.path.splitext(orig)[1].lower()
        is_video = ext in VIDEO_EXTS
        size = 800 if is_hq else 475
        quality = 2 if is_hq else 3
        out = os.path.join(tdir, name)
        gen_thumb(orig, out, size, quality, is_video)

    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(_gen_one, missing)

    print(f"[warmer] Done pre-warming {len(missing)} thumbs.")


def _should_run_background():
    """Only run heavy background tasks in the actual server process, not the reloader parent."""
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true" or "werkzeug" not in str(os.environ.get("SERVER_SOFTWARE", ""))

if _should_run_background():
    threading.Thread(target=_startup_preload, daemon=True).start()


# ─── AI Search (CLIP semantic search + face clusters) ───

_AI_DIR = os.path.join(_APP_DIR, "ai_data")
_ai = {
    "clip_hashes": None, "clip_emb": None, "hash_to_idx": {},
    "model": None, "tokenizer": None, "ready": False,
    "face_clusters": None, "face_index": None,
    "face_embs": None, "face_centroids": None, "emb_to_cluster": None,
    "screenshot_hashes": None,
    "name_to_cluster": {},
}


def _load_ai_index():
    """Load CLIP embeddings + face clusters from disk (no model yet)."""
    import numpy as np

    hashes_path = os.path.join(_AI_DIR, "clip_hashes.json")
    emb_path = os.path.join(_AI_DIR, "clip_embeddings.npy")
    if not os.path.exists(hashes_path) or not os.path.exists(emb_path):
        return

    with open(hashes_path) as f:
        hashes = json.load(f)
    emb = np.load(emb_path)
    _ai["clip_hashes"] = hashes
    _ai["clip_emb"] = emb
    _ai["hash_to_idx"] = {h: i for i, h in enumerate(hashes)}
    print(f"[ai] Loaded {len(hashes)} CLIP embeddings.")

    fc_path = os.path.join(_AI_DIR, "face_clusters.json")
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            _ai["face_clusters"] = json.load(f)
        print(f"[ai] Loaded {len(_ai['face_clusters'])} face clusters.")
        # Clear avatar cache — cluster IDs may have shifted; will regenerate on demand
        avatar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "face_avatars")
        if os.path.isdir(avatar_dir):
            for fn in os.listdir(avatar_dir):
                if fn.endswith(".jpg"):
                    try:
                        os.unlink(os.path.join(avatar_dir, fn))
                    except OSError:
                        pass

    fi_path = os.path.join(_AI_DIR, "face_index.json")
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            _ai["face_index"] = json.load(f)

    # Load face embeddings and build centroid / emb→cluster lookup
    fe_path = os.path.join(_AI_DIR, "face_embeddings.npy")
    if os.path.exists(fe_path) and _ai["face_index"] is not None:
        face_embs = np.load(fe_path)
        _ai["face_embs"] = face_embs
        emb_to_hash = {}
        for ph, faces in _ai["face_index"].items():
            for face in faces:
                emb_to_hash[face["emb_idx"]] = ph
        centroids = {}
        for cid, c in _ai["face_clusters"].items():
            excluded = set(c.get("excluded_hashes", []))
            hset = set(h for h in c.get("photo_hashes", []) if h not in excluded)
            idxs = [idx for idx, h in emb_to_hash.items() if h in hset]
            if idxs:
                centroids[cid] = face_embs[idxs].mean(axis=0)
        _ai["face_centroids"] = centroids
        if centroids:
            cids = list(centroids.keys())
            cmat = np.stack([centroids[c] for c in cids])
            emb_to_cluster = {}
            for emb_idx, ph in emb_to_hash.items():
                dists = np.linalg.norm(cmat - face_embs[emb_idx], axis=1)
                emb_to_cluster[emb_idx] = cids[int(np.argmin(dists))]
            _ai["emb_to_cluster"] = emb_to_cluster
        print(f"[ai] Face embeddings loaded, centroids for {len(centroids)} clusters.")

    # Load screenshot/document hashes
    ss_path = os.path.join(_AI_DIR, "screenshot_hashes.json")
    if os.path.exists(ss_path):
        with open(ss_path) as f:
            _ai["screenshot_hashes"] = set(json.load(f))
        print(f"[ai] Loaded {len(_ai['screenshot_hashes'])} screenshot hashes.")

    _rebuild_name_map()


def _rebuild_name_map():
    """Build lowercase name → cluster_id lookup from face_clusters."""
    clusters = _ai.get("face_clusters") or {}
    mapping = {}
    for cid, c in clusters.items():
        name = (c.get("name") or "").strip()
        if name and not c.get("hidden"):
            mapping[name.lower()] = cid
    _ai["name_to_cluster"] = mapping


def _parse_people_query(q):
    """Split query on ' and ', match parts against known names.

    Returns (matched_people, semantic_text) where matched_people is a list of
    (name, cluster_id) tuples and semantic_text is the remaining non-name text
    (or empty string if everything matched).
    """
    name_map = _ai.get("name_to_cluster") or {}
    parts = [p.strip() for p in q.lower().split(" and ") if p.strip()]
    matched = []
    unmatched = []
    for part in parts:
        if part in name_map:
            matched.append((part, name_map[part]))
        else:
            unmatched.append(part)
    semantic = " and ".join(unmatched)
    return matched, semantic


def _get_person_hashes(cluster_id):
    """Return set of photo_hashes minus excluded_hashes for a cluster."""
    clusters = _ai.get("face_clusters") or {}
    c = clusters.get(cluster_id, {})
    hashes = set(c.get("photo_hashes", []))
    excluded = set(c.get("excluded_hashes", []))
    return hashes - excluded


def _load_clip_model():
    """Load the CLIP text encoder in background for runtime search."""
    if _ai["clip_emb"] is None:
        return
    try:
        import torch
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        _ai["model"] = model
        _ai["tokenizer"] = open_clip.get_tokenizer("ViT-B-32")
        _ai["ready"] = True
        print("[ai] CLIP text encoder loaded — search is ready.")
        # Bust summary cache so covers get re-picked with CLIP aesthetic scoring
        _summary_cache["data"] = None
        _summary_cache["mtime"] = 0
        print("[ai] Summary cache cleared — covers will use CLIP aesthetic scoring.")
    except ImportError:
        print("[ai] torch/open_clip not installed — search disabled.")
    except Exception as e:
        print(f"[ai] Failed to load CLIP model: {e}")


# ─── CLIP aesthetic cover selection ───

_aesthetic_emb = None
_aesthetic_emb_lock = threading.Lock()

AESTHETIC_QUERIES = [
    "beautiful scenic landscape photography",
    "stunning sunset golden hour sky clouds",
    "aesthetic nature photo mountains ocean",
    "travel photography breathtaking view",
    "vibrant outdoor scenery",
    "cinematic landscape photo",
]

PORTRAIT_QUERIES = [
    "a clear well-lit portrait photo of a person smiling",
    "a close-up selfie of a happy person looking at the camera",
    "a nice headshot photo with good lighting",
    "a person posing for the camera with a clear face",
]

_portrait_emb = None
_portrait_emb_lock = threading.Lock()


def _get_portrait_embedding():
    """Compute (once) a CLIP embedding representing 'good portrait photo'."""
    global _portrait_emb
    if _portrait_emb is not None:
        return _portrait_emb
    if not _ai["ready"]:
        return None
    import torch, numpy as np
    with _portrait_emb_lock:
        if _portrait_emb is not None:
            return _portrait_emb
        try:
            tokenizer = _ai["tokenizer"]
            model     = _ai["model"]
            texts = tokenizer(PORTRAIT_QUERIES)
            with torch.no_grad():
                embs = model.encode_text(texts)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                avg  = embs.mean(dim=0)
                avg  = (avg / avg.norm()).cpu().numpy()
            _portrait_emb = avg
            print("[ai] Portrait embedding computed.")
        except Exception as e:
            print(f"[ai] Could not compute portrait embedding: {e}")
    return _portrait_emb


def _get_aesthetic_embedding():
    """Compute (once) a CLIP embedding representing 'aesthetic scenic photo'."""
    global _aesthetic_emb
    if _aesthetic_emb is not None:
        return _aesthetic_emb
    if not _ai["ready"]:
        return None
    import torch, numpy as np
    with _aesthetic_emb_lock:
        if _aesthetic_emb is not None:
            return _aesthetic_emb
        try:
            tokenizer = _ai["tokenizer"]
            model     = _ai["model"]
            texts = tokenizer(AESTHETIC_QUERIES)
            with torch.no_grad():
                embs = model.encode_text(texts)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                avg  = embs.mean(dim=0)
                avg  = (avg / avg.norm()).cpu().numpy()
            _aesthetic_emb = avg
            print("[ai] Aesthetic embedding computed.")
        except Exception as e:
            print(f"[ai] Could not compute aesthetic embedding: {e}")
    return _aesthetic_emb


def _pick_aesthetic_covers(candidates, max_covers=6):
    """Pick cover photos using CLIP aesthetic scoring.

    Scores each candidate against an average 'aesthetic scenic' text embedding,
    then among the top scorers still prefers landscape orientation.
    Falls back to _pick_covers() if CLIP is not ready.
    """
    import numpy as np

    if not _ai["ready"] or _ai["clip_emb"] is None or not candidates:
        return _pick_covers(candidates, max_covers)

    aes_emb = _get_aesthetic_embedding()
    if aes_emb is None:
        return _pick_covers(candidates, max_covers)

    # Score every candidate by cosine similarity to the aesthetic embedding
    scored = []
    for c in candidates:
        thumb = c.get("thumb", "")
        h = thumb.rsplit("/", 1)[-1].replace(".jpg", "")
        idx = _ai["hash_to_idx"].get(h)
        if idx is not None:
            score = float(_ai["clip_emb"][idx] @ aes_emb)
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Take the top 3× pool, then within that prefer landscape orientation
    pool_size = max_covers * 3
    top = [c for _, c in scored[:pool_size]]

    landscape = [c for c in top if _is_landscape(c["thumb"])]
    non_landscape = [c for c in top if not _is_landscape(c["thumb"])]

    import random
    result = landscape[:max_covers]
    if len(result) < max_covers:
        result.extend(non_landscape[:max_covers - len(result)])
    if len(result) < max_covers:
        leftover = [c for _, c in scored[pool_size:]]
        random.shuffle(leftover)
        result.extend(leftover[:max_covers - len(result)])

    random.shuffle(result)
    return [c["thumb"] for c in result[:max_covers]]


def _startup_ai():
    try:
        _load_ai_index()
        _load_clip_model()
    except Exception as e:
        print(f"[ai] Startup error: {e}")


if _should_run_background():
    threading.Thread(target=_startup_ai, daemon=True).start()


@app.route("/api/photos/search")
@require_auth
def api_search_photos():
    """People-aware semantic photo search powered by CLIP + face clusters."""
    import numpy as np

    q = request.args.get("q", "").strip()
    limit = request.args.get("limit", 200, type=int)

    if not q:
        return jsonify({"results": [], "query": ""})

    matched_people, semantic_text = _parse_people_query(q)

    # Build people info for response
    clusters = _ai.get("face_clusters") or {}
    people_info = []
    for name, cid in matched_people:
        c = clusters.get(cid, {})
        people_info.append({
            "name": c.get("name", name),
            "cluster_id": cid,
            "photo_count": c.get("photo_count", 0),
            "sample_face": c.get("sample_face", ""),
        })

    # Build hash→item lookup
    items = load_photo_index()
    hash_to_item = {}
    for item in items:
        url = item.get("thumb", "")
        if url:
            h = url.rsplit("/", 1)[-1].replace(".jpg", "")
            hash_to_item[h] = item

    # Case 1: People only (no semantic text)
    if matched_people and not semantic_text:
        # Intersect photo sets from all matched people
        person_sets = [_get_person_hashes(cid) for _, cid in matched_people]
        combined = person_sets[0]
        for s in person_sets[1:]:
            combined = combined & s

        results = []
        for h in combined:
            item = hash_to_item.get(h)
            if item:
                results.append(item)
        results.sort(key=lambda x: -x.get("date", 0))
        results = results[:limit]

        return jsonify({
            "results": results, "query": q,
            "people": people_info, "sort": "date",
        })

    # Case 2: People + semantic text (CLIP-score only that person's photos)
    # Case 3: Pure semantic (no people matched)
    if _ai["clip_emb"] is None:
        return jsonify({"error": "AI index not built. SSH into the NAS and run: python ai_indexer.py"}), 404
    if not _ai["ready"]:
        return jsonify({"error": "AI search is loading, try again in a moment..."}), 503

    import torch
    model = _ai["model"]
    tokenizer = _ai["tokenizer"]

    clip_query = semantic_text if semantic_text else q
    text = tokenizer([clip_query])
    with torch.no_grad():
        tf = model.encode_text(text)
        tf = (tf / tf.norm(dim=-1, keepdim=True)).squeeze().cpu().numpy()

    scores = _ai["clip_emb"] @ tf

    if matched_people:
        # Intersect person sets, then rank by CLIP score within that subset
        person_sets = [_get_person_hashes(cid) for _, cid in matched_people]
        allowed = person_sets[0]
        for s in person_sets[1:]:
            allowed = allowed & s

        scored = []
        for i, h in enumerate(_ai["clip_hashes"]):
            if h in allowed:
                scored.append((i, float(scores[i])))
        scored.sort(key=lambda x: -x[1])
        scored = scored[:limit]

        results = []
        for idx, score in scored:
            h = _ai["clip_hashes"][idx]
            item = hash_to_item.get(h)
            if item and score > 0.15:
                results.append({**item, "score": round(score, 3)})

        return jsonify({
            "results": results, "query": q,
            "people": people_info, "sort": "relevance",
        })

    # Pure semantic search (no people)
    top_idx = np.argsort(scores)[::-1][:limit]
    results = []
    for idx in top_idx:
        h = _ai["clip_hashes"][idx]
        item = hash_to_item.get(h)
        score = float(scores[idx])
        if item and score > 0.18:
            results.append({**item, "score": round(score, 3)})

    return jsonify({
        "results": results, "query": q,
        "people": [], "sort": "relevance",
    })


@app.route("/api/photos/search/status")
@require_auth
def api_search_status():
    """Check if AI search is available."""
    has_index = _ai["clip_emb"] is not None
    count = len(_ai["clip_hashes"]) if _ai["clip_hashes"] else 0
    faces = len(_ai["face_clusters"]) if _ai["face_clusters"] else 0
    return jsonify({
        "indexed": has_index,
        "ready": _ai["ready"],
        "count": count,
        "faces": faces,
    })


@app.route("/api/photos/people/names")
@require_auth
def api_people_names():
    """Return named, non-hidden people for search autocomplete."""
    clusters = _ai.get("face_clusters") or {}
    result = []
    for cid, c in clusters.items():
        name = (c.get("name") or "").strip()
        if name and not c.get("hidden"):
            result.append({
                "name": name,
                "cluster_id": cid,
                "photo_count": c.get("photo_count", 0),
                "sample_face": c.get("sample_face", ""),
            })
    result.sort(key=lambda x: x["name"].lower())
    return jsonify(result)


@app.route("/api/photos/people/create", methods=["POST"])
@require_auth
def api_people_create():
    """Create a new person cluster from seed photos. Expands to similar faces automatically."""
    data = request.json or {}
    name = (data.get("name") or "").strip()
    seed_hashes = list(data.get("photo_hashes", []))

    if not name:
        return jsonify({"error": "Name required"}), 400
    if not seed_hashes:
        return jsonify({"error": "No photos selected"}), 400

    face_index = _ai.get("face_index")
    face_embs = _ai.get("face_embs")
    clusters = _ai.get("face_clusters") or {}

    if face_index is None or face_embs is None or len(face_embs) == 0:
        return jsonify({"error": "Face data not loaded — run ai_indexer.py --rescan-faces first"}), 503

    # Check for duplicate name
    for c in clusters.values():
        if (c.get("name") or "").lower() == name.lower():
            return jsonify({"error": f"'{name}' already exists"}), 409

    # Collect face embeddings from seed photos
    seed_emb_indices = []
    emb_to_hash = {}
    for ph, faces in face_index.items():
        for face in faces:
            emb_to_hash[face["emb_idx"]] = ph
    for h in seed_hashes:
        for face in face_index.get(h, []):
            idx = face["emb_idx"]
            if idx < len(face_embs):
                seed_emb_indices.append(idx)

    if not seed_emb_indices:
        return jsonify({"error": "No faces detected in the selected photos"}), 400

    import numpy as np
    seed_arr = np.array(seed_emb_indices)

    # Refine centroid: seed photos may contain other people's faces (group photos).
    # Iteratively drop embeddings that are far from the centroid so we converge
    # on just this person's face cluster.
    active = seed_arr.copy()
    for _ in range(5):
        centroid = face_embs[active].mean(axis=0)
        dists_seed = np.linalg.norm(face_embs[active] - centroid, axis=1)
        cutoff = dists_seed.mean() + dists_seed.std()
        filtered = active[dists_seed <= cutoff]
        if len(filtered) < max(3, len(active) * 0.5):
            break  # don't over-prune
        if len(filtered) == len(active):
            break  # converged
        active = filtered

    centroid = face_embs[active].mean(axis=0)

    # Expand: assign all embeddings within threshold to this person
    EXPAND_THRESH = 0.75
    assigned = set(seed_hashes)
    dists = np.linalg.norm(face_embs - centroid, axis=1)
    for emb_idx, dist in enumerate(dists):
        if dist < EXPAND_THRESH:
            ph = emb_to_hash.get(emb_idx)
            if ph:
                assigned.add(ph)

    # Find best sample face (seed embedding closest to centroid)
    best_hash = seed_hashes[0]
    best_dist = float("inf")
    for idx in seed_emb_indices:
        d = float(np.linalg.norm(face_embs[idx] - centroid))
        if d < best_dist:
            best_dist = d
            best_hash = emb_to_hash.get(idx, seed_hashes[0])

    new_id = str(max((int(k) for k in clusters.keys()), default=-1) + 1)
    photo_hashes = sorted(assigned)
    clusters[new_id] = {
        "name": name,
        "photo_count": len(photo_hashes),
        "face_count": len(seed_emb_indices),
        "sample_face": best_hash,
        "photo_hashes": photo_hashes,
        "excluded_hashes": [],
    }

    fc_path = os.path.join(_AI_DIR, "face_clusters.json")
    with open(fc_path, "w") as f:
        json.dump(clusters, f, indent=2)
    _ai["face_clusters"] = clusters
    _rebuild_name_map()

    return jsonify({"status": "ok", "id": new_id, "name": name, "photo_count": len(photo_hashes)})


@app.route("/api/photos/faces")
@require_auth
def api_faces():
    """Return face clusters for people browsing."""
    clusters = _ai.get("face_clusters")
    ss = _get_screenshot_hashes()
    if not clusters:
        return jsonify({"clusters": [], "screenshot_count": len(ss)})

    result = []
    for cid, c in clusters.items():
        if c.get("hidden"):
            continue
        result.append({
            "id": cid,
            "name": c.get("name", ""),
            "photo_count": c.get("photo_count", 0),
            "face_count": c.get("face_count", 0),
            "sample_face": c.get("sample_face", 0),
        })
    return jsonify({"clusters": result, "screenshot_count": len(ss)})


@app.route("/api/photos/screenshots")
@require_auth
def api_screenshots():
    """Return all photos classified as screenshots/documents."""
    ss_hashes = _get_screenshot_hashes()
    if not ss_hashes:
        return jsonify([])
    items = load_photo_index()
    results = []
    for item in items:
        url = item.get("thumb", "")
        if url:
            h = url.rsplit("/", 1)[-1].replace(".jpg", "")
            if h in ss_hashes:
                results.append(item)
    results.sort(key=lambda x: -x.get("date", 0))
    return jsonify(results)


@app.route("/api/photos/face/<cluster_id>")
@require_auth
def api_face_photos(cluster_id):
    """Return all photos for a specific face cluster."""
    clusters = _ai.get("face_clusters")
    if not clusters or cluster_id not in clusters:
        return jsonify([])

    photo_hashes = set(clusters[cluster_id].get("photo_hashes", []))
    new_hashes = set(clusters[cluster_id].get("new_hashes", []))
    items = load_photo_index()
    results = []
    for item in items:
        url = item.get("thumb", "")
        if url:
            h = url.rsplit("/", 1)[-1].replace(".jpg", "")
            if h in photo_hashes:
                item_copy = dict(item)
                if h in new_hashes:
                    item_copy["is_new"] = True
                results.append(item_copy)
    # New photos first, then by date
    results.sort(key=lambda x: (0 if x.get("is_new") else 1, -x.get("date", 0)))
    return jsonify(results)


@app.route("/api/photos/face/<cluster_id>/name", methods=["POST"])
@require_auth
def api_name_face(cluster_id):
    """Set a display name for a face cluster."""
    clusters = _ai.get("face_clusters")
    if not clusters or cluster_id not in clusters:
        return jsonify({"error": "Cluster not found"}), 404

    name = request.json.get("name", "").strip()
    clusters[cluster_id]["name"] = name

    fc_path = os.path.join(_AI_DIR, "face_clusters.json")
    with open(fc_path, "w") as f:
        json.dump(clusters, f, indent=2)
    _rebuild_name_map()

    return jsonify({"status": "ok", "name": name})


@app.route("/api/photos/faces/merge", methods=["POST"])
@require_auth
def api_merge_faces():
    """Merge two face clusters (source into target). Keeps target's name."""
    clusters = _ai.get("face_clusters")
    if not clusters:
        return jsonify({"error": "No face data"}), 404

    source_id = str(request.json.get("source", ""))
    target_id = str(request.json.get("target", ""))

    if source_id not in clusters or target_id not in clusters:
        return jsonify({"error": "Cluster not found"}), 404
    if source_id == target_id:
        return jsonify({"error": "Cannot merge cluster with itself"}), 400

    src = clusters[source_id]
    tgt = clusters[target_id]

    merged_hashes = list(set(tgt.get("photo_hashes", []) + src.get("photo_hashes", [])))
    tgt["photo_hashes"] = merged_hashes
    tgt["photo_count"] = len(merged_hashes)
    tgt["face_count"] = tgt.get("face_count", 0) + src.get("face_count", 0)
    if not tgt.get("name") and src.get("name"):
        tgt["name"] = src["name"]

    del clusters[source_id]

    fc_path = os.path.join(_AI_DIR, "face_clusters.json")
    with open(fc_path, "w") as f:
        json.dump(clusters, f, indent=2)
    _rebuild_name_map()
    _invalidate_avatar(source_id)
    _invalidate_avatar(target_id)

    return jsonify({
        "status": "ok",
        "target_id": target_id,
        "name": tgt["name"],
        "photo_count": tgt["photo_count"],
    })


@app.route("/api/photos/face/<cluster_id>/remove", methods=["POST"])
@require_auth
def api_remove_from_face(cluster_id):
    """Remove photos from a face cluster (they don't belong to this person)."""
    clusters = _ai.get("face_clusters")
    if not clusters or cluster_id not in clusters:
        return jsonify({"error": "Cluster not found"}), 404

    hashes_to_remove = set(request.json.get("hashes", []))
    if not hashes_to_remove:
        return jsonify({"error": "No hashes provided"}), 400

    c = clusters[cluster_id]
    before = len(c.get("photo_hashes", []))
    c["photo_hashes"] = [h for h in c["photo_hashes"] if h not in hashes_to_remove]
    c["photo_count"] = len(c["photo_hashes"])
    removed = before - c["photo_count"]

    # Persist excluded hashes so re-clustering doesn't bring them back
    excluded = set(c.get("excluded_hashes", []))
    excluded.update(hashes_to_remove)
    c["excluded_hashes"] = list(excluded)

    fc_path = os.path.join(_AI_DIR, "face_clusters.json")
    with open(fc_path, "w") as f:
        json.dump(clusters, f, indent=2)
    _invalidate_avatar(cluster_id)

    return jsonify({
        "status": "ok",
        "removed": removed,
        "remaining": c["photo_count"],
    })


@app.route("/api/photos/faces/suggestions")
@require_auth
def api_face_suggestions():
    """Return merge suggestions: pairs of clusters that might be the same person."""
    import numpy as np

    clusters = _ai.get("face_clusters")
    if not clusters:
        return jsonify({"suggestions": []})

    emb_path = os.path.join(_AI_DIR, "face_embeddings.npy")
    fi_path = os.path.join(_AI_DIR, "face_index.json")
    if not os.path.exists(emb_path) or not os.path.exists(fi_path):
        return jsonify({"suggestions": []})

    embs = np.load(emb_path)
    with open(fi_path) as f:
        face_data = json.load(f)

    emb_to_hash = {}
    for photo_hash, faces in face_data.items():
        for face in faces:
            emb_to_hash[face["emb_idx"]] = photo_hash

    # Compute centroids for each cluster
    centroids = {}
    for cid, c in clusters.items():
        hset = set(c.get("photo_hashes", []))
        idxs = [idx for idx, h in emb_to_hash.items() if h in hset]
        if idxs:
            centroids[cid] = embs[idxs].mean(axis=0)

    cids = list(centroids.keys())
    suggestions = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a, b = clusters[cids[i]], clusters[cids[j]]
            na, nb = a.get("name", ""), b.get("name", "")
            # Skip if both are named differently -- obviously different people
            if na and nb and na != nb:
                continue
            dist = float(np.linalg.norm(centroids[cids[i]] - centroids[cids[j]]))
            if dist < 0.45:
                suggestions.append({
                    "cluster_a": cids[i],
                    "cluster_b": cids[j],
                    "name_a": na,
                    "name_b": nb,
                    "count_a": a.get("photo_count", 0),
                    "count_b": b.get("photo_count", 0),
                    "face_a": a.get("sample_face", 0),
                    "face_b": b.get("sample_face", 0),
                    "distance": round(dist, 3),
                })
    suggestions.sort(key=lambda x: x["distance"])
    return jsonify({"suggestions": suggestions[:30]})



@app.route("/api/photos/faces/in-photo")
@require_auth
def api_faces_in_photo():
    """Return all detected faces in a photo with their cluster assignments."""
    photo_hash = request.args.get("hash", "")
    face_index = _ai.get("face_index")
    clusters = _ai.get("face_clusters")
    emb_to_cluster = _ai.get("emb_to_cluster")

    if not face_index or not clusters:
        return jsonify([])

    faces = face_index.get(photo_hash, [])
    result = []
    for face in faces:
        emb_idx = face["emb_idx"]
        cid = emb_to_cluster.get(emb_idx) if emb_to_cluster else None
        c = clusters.get(cid, {}) if cid else {}
        result.append({
            "emb_idx": emb_idx,
            "bbox": face["bbox"],
            "crop_url": f"/static/faces/{emb_idx}.jpg",
            "cluster_id": cid,
            "name": c.get("name") or "Unknown",
        })
    return jsonify(result)


@app.route("/api/photos/people/in-photo")
@require_auth
def api_people_in_photo():
    """Return named people in a photo by cluster membership lookup."""
    photo_hash = request.args.get("hash", "")
    clusters = _ai.get("face_clusters") or {}
    if not photo_hash or not clusters:
        return jsonify([])
    result = []
    for cid, c in clusters.items():
        if c.get("hidden"):
            continue
        name = (c.get("name") or "").strip()
        if not name:
            continue
        excluded = set(c.get("excluded_hashes", []))
        if photo_hash in set(c.get("photo_hashes", [])) and photo_hash not in excluded:
            result.append({
                "name": name,
                "cluster_id": cid,
                "sample_face": c.get("sample_face", ""),
            })
    return jsonify(result)


def _square_face_crop(thumb_path, bbox, size=200):
    """Crop a square region centered on a face from a thumbnail."""
    from PIL import Image
    import io
    img = Image.open(thumb_path).convert("RGB")
    tw, th = img.size
    top, right, bottom, left = bbox
    # Scale bbox if detected on 1024px original, not thumbnail
    if right > tw or bottom > th:
        if tw >= th:
            ow, oh = 1024, round(th * 1024 / tw)
        else:
            oh, ow = 1024, round(tw * 1024 / th)
        sx, sy = tw / ow, th / oh
        top, bottom = int(top * sy), int(bottom * sy)
        left, right = int(left * sx), int(right * sx)
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    side = max(right - left, bottom - top) * 1.6  # face + context
    # Shrink side if it can't fit in the image at all
    side = min(side, tw, th)
    half = side / 2
    # Clamp center so the square stays within image bounds
    cx = max(half, min(tw - half, cx))
    cy = max(half, min(th - half, cy))
    x1, y1 = int(cx - half), int(cy - half)
    x2, y2 = int(cx + half), int(cy + half)
    crop = img.crop((x1, y1, x2, y2)).resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=88)
    buf.seek(0)
    return buf


_AVATAR_DIR = os.path.join(_APP_DIR, "static", "face_avatars")
os.makedirs(_AVATAR_DIR, exist_ok=True)


def _invalidate_avatar(cluster_id):
    """Delete cached avatar for a cluster so it gets regenerated."""
    import glob as _glob
    for f in _glob.glob(os.path.join(_AVATAR_DIR, f"{cluster_id}_v*.jpg")):
        try:
            os.unlink(f)
        except OSError:
            pass


@app.route("/api/photos/people/<cluster_id>/avatar", methods=["GET", "POST"])
@require_auth
def api_person_avatar(cluster_id):
    """GET: serve avatar. POST: set manual avatar from a photo hash."""
    import numpy as np
    from PIL import Image
    import io

    clusters = _ai.get("face_clusters") or {}
    face_index = _ai.get("face_index")
    c = clusters.get(cluster_id)
    if not c:
        return jsonify({"error": "Not found"}), 404

    # --- POST: set manual avatar ---
    if request.method == "POST":
        data = request.json or {}
        photo_hash = data.get("hash")
        if not photo_hash or not face_index:
            return jsonify({"error": "Invalid request"}), 400

        emb_to_cluster = _ai.get("emb_to_cluster") or {}
        face_embs = _ai.get("face_embs")
        centroids = _ai.get("face_centroids") or {}
        centroid = centroids.get(cluster_id)
        faces = face_index.get(photo_hash, [])

        # Find this person's face — try emb_to_cluster, then nearest to centroid
        best_bbox = None
        best_size = 0
        for face in faces:
            if emb_to_cluster.get(face["emb_idx"]) == cluster_id:
                top, right, bottom, left = face["bbox"]
                sz = max(right - left, bottom - top)
                if sz > best_size:
                    best_size = sz
                    best_bbox = face["bbox"]

        # Fallback: pick the face closest to this person's centroid
        if not best_bbox and centroid is not None and face_embs is not None:
            best_dist = 999.0
            for face in faces:
                idx = face["emb_idx"]
                if idx < len(face_embs):
                    dist = float(np.linalg.norm(face_embs[idx] - centroid))
                    if dist < best_dist:
                        best_dist = dist
                        best_bbox = face["bbox"]

        # Last resort: largest face
        if not best_bbox:
            best_size = 0
            for face in faces:
                top, right, bottom, left = face["bbox"]
                sz = max(right - left, bottom - top)
                if sz > best_size:
                    best_size = sz
                    best_bbox = face["bbox"]

        chosen_bbox = best_bbox
        if not chosen_bbox:
            # No detected face at all — center crop the thumbnail
            thumb_path = os.path.join(app.static_folder, "thumbs", photo_hash + ".jpg")
            if not os.path.exists(thumb_path):
                return jsonify({"error": "Photo not found"}), 404
            c["avatar_hash"] = photo_hash
            c["avatar_bbox"] = None
            fc_path = os.path.join(_AI_DIR, "face_clusters.json")
            with open(fc_path, "w") as f:
                json.dump(clusters, f, indent=2)
            _invalidate_avatar(cluster_id)
            try:
                img = Image.open(thumb_path).convert("RGB")
                w, h = img.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                sq = img.crop((left, top, left + side, top + side))
                sq = sq.resize((300, 300), Image.LANCZOS)
                buf = io.BytesIO()
                sq.save(buf, "JPEG", quality=90)
                buf.seek(0)
                v = request.args.get("v", "11")
                cache_path = os.path.join(_AVATAR_DIR, f"{cluster_id}_v{v}_{photo_hash[:8]}.jpg")
                with open(cache_path, "wb") as fout:
                    fout.write(buf.read())
            except Exception:
                pass
            return jsonify({"status": "ok"})

        # Save the override
        c["avatar_hash"] = photo_hash
        c["avatar_bbox"] = list(chosen_bbox)
        fc_path = os.path.join(_AI_DIR, "face_clusters.json")
        with open(fc_path, "w") as f:
            json.dump(clusters, f, indent=2)

        # Generate and cache the avatar crop
        _invalidate_avatar(cluster_id)
        thumb_path = os.path.join(app.static_folder, "thumbs", photo_hash + ".jpg")
        if os.path.exists(thumb_path):
            v = request.args.get("v", "11")
            cache_path = os.path.join(_AVATAR_DIR, f"{cluster_id}_v{v}_{photo_hash[:8]}.jpg")
            try:
                buf = _square_face_crop(thumb_path, chosen_bbox, size=300)
                data_bytes = buf.read()
                with open(cache_path, "wb") as fout:
                    fout.write(data_bytes)
            except Exception:
                pass

        return jsonify({"status": "ok"})

    # --- GET: serve avatar ---

    v = request.args.get("v", "11")
    avatar_hash = c.get("avatar_hash")

    # Cache key includes avatar_hash so manual PFPs never get stale algorithmic cache
    if avatar_hash:
        cache_path = os.path.join(_AVATAR_DIR, f"{cluster_id}_v{v}_{avatar_hash[:8]}.jpg")
    else:
        cache_path = os.path.join(_AVATAR_DIR, f"{cluster_id}_v{v}.jpg")

    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype="image/jpeg",
                         max_age=86400, conditional=True)

    # Manual avatar override — use the pinned photo + stored bbox
    if avatar_hash:
        thumb_path = os.path.join(app.static_folder, "thumbs", avatar_hash + ".jpg")
        if os.path.exists(thumb_path):
            # Use stored bbox, or find face via emb_to_cluster
            avatar_bbox = c.get("avatar_bbox")
            if not avatar_bbox and face_index:
                emb_to_cluster = _ai.get("emb_to_cluster") or {}
                faces = face_index.get(avatar_hash, [])
                best_sz = 0
                for face in faces:
                    top, right, bottom, left = face["bbox"]
                    sz = max(right - left, bottom - top)
                    if emb_to_cluster.get(face["emb_idx"]) == cluster_id and sz > best_sz:
                        best_sz = sz
                        avatar_bbox = face["bbox"]
                # Fallback: largest face in photo
                if not avatar_bbox:
                    for face in faces:
                        top, right, bottom, left = face["bbox"]
                        sz = max(right - left, bottom - top)
                        if sz > best_sz:
                            best_sz = sz
                            avatar_bbox = face["bbox"]
            if avatar_bbox:
                try:
                    buf = _square_face_crop(thumb_path, avatar_bbox, size=300)
                    data_bytes = buf.read()
                    with open(cache_path, "wb") as fout:
                        fout.write(data_bytes)
                    return send_file(cache_path, mimetype="image/jpeg",
                                     max_age=86400, conditional=True)
                except Exception:
                    pass

    if not face_index:
        return jsonify({"error": "Not found"}), 404

    face_embs = _ai.get("face_embs")
    centroids = _ai.get("face_centroids") or {}
    emb_to_cluster = _ai.get("emb_to_cluster") or {}
    centroid = centroids.get(cluster_id)
    excluded = set(c.get("excluded_hashes", []))
    photo_hashes = [h for h in c.get("photo_hashes", []) if h not in excluded]

    candidates = []
    for ph in photo_hashes:
        faces = face_index.get(ph, [])
        best_size = 0
        best_dist = 999.0
        best_bbox = None
        biggest_other = 0
        for face in faces:
            idx = face["emb_idx"]
            top, right, bottom, left = face["bbox"]
            face_size = max(right - left, bottom - top)
            if emb_to_cluster.get(idx) != cluster_id:
                # Track the largest *other* person's face
                if face_size > biggest_other:
                    biggest_other = face_size
                continue
            if face_size > best_size:
                best_size = face_size
                best_bbox = face["bbox"]
                if centroid is not None and face_embs is not None and idx < len(face_embs):
                    best_dist = float(np.linalg.norm(face_embs[idx] - centroid))

        # Require a decently large face — reject tiny detections
        if best_size < 80 or best_bbox is None:
            continue

        # Face dominance: ratio of our face to the biggest other face
        # 1.0+ means we're the biggest face, <1.0 means someone else dominates
        dominance = best_size / biggest_other if biggest_other > 0 else 5.0

        candidates.append({
            "hash": ph,
            "bbox": best_bbox,
            "face_size": best_size,
            "face_dist": best_dist,
            "num_faces": len(faces),
            "dominance": dominance,
        })

    if not candidates:
        return jsonify({"error": "No face found"}), 404

    # Pre-sort by face size and only quality-score the top 50
    candidates.sort(key=lambda c: -c["face_size"])
    candidates = candidates[:50]

    # Score candidates using actual image quality of the face crop
    scored = []
    for cand in candidates:
        ph = cand["hash"]
        thumb_path = os.path.join(app.static_folder, "thumbs", ph + ".jpg")
        if not os.path.exists(thumb_path):
            continue
        try:
            img = Image.open(thumb_path).convert("RGB")
        except Exception:
            continue

        tw, th = img.size
        top, right, bottom, left = cand["bbox"]
        # Scale bbox if detected on 1024px original
        if right > tw or bottom > th:
            if tw >= th:
                ow, oh = 1024, round(th * 1024 / tw)
            else:
                oh, ow = 1024, round(tw * 1024 / th)
            sx, sy = tw / ow, th / oh
            top, bottom = int(top * sy), int(bottom * sy)
            left, right = int(left * sx), int(right * sx)

        fw, fh = right - left, bottom - top
        face_size = max(fw, fh)

        # Crop the face region for quality checks
        cx, cy = (left + right) / 2, (top + bottom) / 2
        side = max(fw, fh) * 1.4
        half = side / 2
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(tw, int(cx + half))
        y2 = min(th, int(cy + half))
        face_crop = img.crop((x1, y1, x2, y2))

        # Sharpness — Laplacian variance via numpy (fast)
        fc_small = face_crop.convert("L").resize((64, 64), Image.LANCZOS)
        arr = np.array(fc_small, dtype=np.float32)
        # Laplacian approximation: variance of difference from neighbors
        lap = (4 * arr[1:-1, 1:-1]
               - arr[:-2, 1:-1] - arr[2:, 1:-1]
               - arr[1:-1, :-2] - arr[1:-1, 2:])
        sharpness = float(np.var(lap))

        # Brightness — reject too dark or blown out
        avg_brightness = float(arr.mean())
        brightness_ok = 1.0 if 60 < avg_brightness < 220 else 0.3

        # Face-to-image ratio — bigger face in frame = better portrait
        face_ratio = face_size / max(tw, th)

        # Identity closeness (how close to cluster centroid)
        identity = max(0.0, 1.0 - min(cand["face_dist"], 2.0))

        # Fewer faces in photo = more likely a portrait
        solo_bonus = 1.0 if cand["num_faces"] <= 2 else 0.6

        # Face dominance — is this person the star of the photo?
        # If someone else's face is bigger, heavily penalize
        dom = cand["dominance"]
        if dom < 0.8:
            dominance_mult = 0.2    # someone else dominates — bad avatar
        elif dom < 1.2:
            dominance_mult = 0.7    # roughly equal — meh
        else:
            dominance_mult = 1.0    # we're the biggest face — good

        # Composite score — face size, sharpness, and dominance rule
        score = (
            face_size * 3.0 +              # big face = good
            sharpness * 0.005 +             # sharp = good
            face_ratio * 500.0 +            # face fills frame = good
            identity * 80.0 +               # looks like this person = good
            solo_bonus * 100.0              # solo/duo photo = good
        ) * brightness_ok * dominance_mult  # dark/dominated = bad

        scored.append((score, ph, cand["bbox"]))

    if not scored:
        return jsonify({"error": "No face found"}), 404

    scored.sort(key=lambda x: -x[0])

    # Crop around the actual face
    for _, best_hash, bbox in scored[:10]:
        thumb_path = os.path.join(app.static_folder, "thumbs", best_hash + ".jpg")
        if not os.path.exists(thumb_path):
            continue
        try:
            buf = _square_face_crop(thumb_path, bbox, size=300)
            data = buf.read()
            with open(cache_path, "wb") as f:
                f.write(data)
            return send_file(cache_path, mimetype="image/jpeg",
                             max_age=86400, conditional=True)
        except Exception:
            continue

    return jsonify({"error": "No usable face found"}), 404


@app.route("/api/photos/face-crop/<photo_hash>/<int:emb_idx>")
@require_auth
def api_face_crop(photo_hash, emb_idx):
    """Serve a cropped face image with padding."""
    face_index = _ai.get("face_index")
    if not face_index:
        return jsonify({"error": "No face index"}), 404

    faces = face_index.get(photo_hash, [])
    face = next((f for f in faces if f["emb_idx"] == emb_idx), None)
    if not face:
        return jsonify({"error": "Face not found"}), 404

    thumb_path = os.path.join(app.static_folder, "thumbs", photo_hash + ".jpg")
    if not os.path.exists(thumb_path):
        return jsonify({"error": "Photo not found"}), 404

    try:
        buf = _square_face_crop(thumb_path, face["bbox"], size=180)
        return send_file(buf, mimetype="image/jpeg",
                         max_age=86400, conditional=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/photos/face/move", methods=["POST"])
@require_auth
def api_move_face():
    """Move a photo from one face cluster to another."""
    data = request.json or {}
    photo_hash = data.get("hash")
    from_id = data.get("from_cluster")
    to_id = data.get("to_cluster")

    clusters = _ai.get("face_clusters")
    if not clusters:
        return jsonify({"error": "No face clusters"}), 404
    if not photo_hash or from_id not in clusters or to_id not in clusters:
        return jsonify({"error": "Invalid request"}), 400
    if from_id == to_id:
        return jsonify({"status": "ok", "moved": False})

    from_c = clusters[from_id]
    to_c = clusters[to_id]

    # Remove from source + add to excluded so re-cluster doesn't restore it
    if photo_hash in from_c.get("photo_hashes", []):
        from_c["photo_hashes"] = [h for h in from_c["photo_hashes"] if h != photo_hash]
        from_c["photo_count"] = len(from_c["photo_hashes"])
        excl = set(from_c.get("excluded_hashes", []))
        excl.add(photo_hash)
        from_c["excluded_hashes"] = list(excl)

    # Add to destination (avoid duplicates)
    if photo_hash not in to_c.get("photo_hashes", []):
        to_c.setdefault("photo_hashes", []).append(photo_hash)
        to_c["photo_count"] = len(to_c["photo_hashes"])
        # Un-exclude from destination if it was previously removed
        excl_to = set(to_c.get("excluded_hashes", []))
        if photo_hash in excl_to:
            excl_to.discard(photo_hash)
            if excl_to:
                to_c["excluded_hashes"] = list(excl_to)
            else:
                to_c.pop("excluded_hashes", None)

    fc_path = os.path.join(_AI_DIR, "face_clusters.json")
    with open(fc_path, "w") as f:
        json.dump(clusters, f, indent=2)
    _invalidate_avatar(from_id)
    _invalidate_avatar(to_id)

    return jsonify({"status": "ok", "moved": True,
                    "from_count": from_c["photo_count"],
                    "to_count": to_c["photo_count"]})


@app.route("/api/photos/face/review")
@require_auth
def api_face_review():
    """Return a batch of ambiguous faces for user review (Google Photos style)."""
    clusters = _ai.get("face_clusters")
    face_index = _ai.get("face_index")
    embs = _ai.get("face_embs")
    if not clusters or face_index is None or embs is None:
        return jsonify({"items": [], "remaining": 0})

    named = {cid: c for cid, c in clusters.items() if c.get("name")}
    if not named:
        return jsonify({"items": [], "remaining": 0})

    assigned = set()
    for c in clusters.values():
        assigned.update(c.get("photo_hashes", []))

    # Build per-cluster excluded sets so we skip already-rejected pairs
    cluster_excluded = {}
    for cid, c in clusters.items():
        excl = c.get("excluded_hashes")
        if excl:
            cluster_excluded[cid] = set(excl)

    emb_to_hash = {}
    for ph, faces in face_index.items():
        for face in faces:
            emb_to_hash[face["emb_idx"]] = ph

    # Compute centroids
    centroid_cids, centroid_vecs = [], []
    for cid, c in named.items():
        excluded = set(c.get("excluded_hashes", []))
        ph_set = set(c.get("photo_hashes", [])) - excluded
        idxs = [idx for idx, ph in emb_to_hash.items() if ph in ph_set and idx < len(embs)]
        if idxs:
            centroid_cids.append(cid)
            centroid_vecs.append(embs[idxs].mean(axis=0))

    if not centroid_vecs:
        return jsonify({"items": [], "remaining": 0})

    import numpy as _np
    centroid_mat = _np.array(centroid_vecs)

    THRESH = 1.05
    items = load_photo_index()
    hash_to_item = {}
    for item in items:
        url = item.get("thumb", "")
        if url:
            h = url.rsplit("/", 1)[-1].replace(".jpg", "")
            hash_to_item[h] = item

    candidates = []
    for emb_idx, ph in emb_to_hash.items():
        if ph in assigned or emb_idx >= len(embs):
            continue
        dists = _np.linalg.norm(centroid_mat - embs[emb_idx], axis=1)
        order = _np.argsort(dists)
        # Walk down candidates until we find one not excluded
        best_cid = None
        best_d = None
        second_d = 999
        for rank, oi in enumerate(order):
            cid = centroid_cids[int(oi)]
            d = float(dists[oi])
            excl = cluster_excluded.get(cid)
            if excl and ph in excl:
                continue  # already rejected for this person
            if best_cid is None:
                best_cid = cid
                best_d = d
            elif second_d == 999:
                second_d = d
                break
        if best_cid is None or best_d >= THRESH:
            continue
        candidates.append({
            "photo_hash": ph,
            "emb_idx": emb_idx,
            "best_cid": best_cid,
            "best_name": named[best_cid].get("name", ""),
            "best_dist": best_d,
            "margin": second_d - best_d,
            "sample_face": named[best_cid].get("sample_face", ""),
        })

    # Sort: highest confidence first (lowest distance, highest margin)
    candidates.sort(key=lambda x: (x["best_dist"] - x["margin"] * 2))

    # Skip already-reviewed photos (sent from frontend)
    skip_param = request.args.get("skip", "")
    skip_hashes = set(skip_param.split(",")) if skip_param else set()

    # Dedupe by photo hash (one review per photo)
    seen = set()
    unique = []
    for c in candidates:
        if c["photo_hash"] not in seen and c["photo_hash"] not in skip_hashes:
            seen.add(c["photo_hash"])
            unique.append(c)

    batch_size = int(request.args.get("batch", 20))
    batch = unique[:batch_size]

    result = []
    for c in batch:
        item = hash_to_item.get(c["photo_hash"])
        if not item:
            continue
        result.append({
            "photo_hash": c["photo_hash"],
            "thumb": item.get("thumb_hq") or item.get("thumb", ""),
            "path": item.get("path", ""),
            "suggested_cid": c["best_cid"],
            "suggested_name": c["best_name"],
            "suggested_avatar": c["sample_face"],
            "confidence": round(max(0, min(100, (1.05 - c["best_dist"]) / 1.05 * 100)), 1),
            "margin": round(c["margin"], 3),
        })

    return jsonify({"items": result, "remaining": len(unique)})


@app.route("/api/photos/face/review", methods=["POST"])
@require_auth
def api_face_review_submit():
    """Accept or reject a face review suggestion."""
    data = request.json or {}
    photo_hash = data.get("photo_hash")
    cluster_id = data.get("cluster_id")
    accept = data.get("accept", False)

    clusters = _ai.get("face_clusters")
    if not clusters or not photo_hash:
        return jsonify({"error": "Invalid request"}), 400

    if accept and cluster_id in clusters:
        c = clusters[cluster_id]
        if photo_hash not in c.get("photo_hashes", []):
            c.setdefault("photo_hashes", []).append(photo_hash)
            c["photo_count"] = len(c["photo_hashes"])
        # Un-exclude if previously excluded
        excl = set(c.get("excluded_hashes", []))
        if photo_hash in excl:
            excl.discard(photo_hash)
            c["excluded_hashes"] = list(excl) if excl else []
    elif not accept and cluster_id in clusters:
        # Mark as excluded from this cluster so it doesn't get suggested again
        c = clusters[cluster_id]
        excl = set(c.get("excluded_hashes", []))
        excl.add(photo_hash)
        c["excluded_hashes"] = list(excl)

    fc_path = os.path.join(_AI_DIR, "face_clusters.json")
    with open(fc_path, "w") as f:
        json.dump(clusters, f, indent=2)

    return jsonify({"status": "ok"})


def _run_startup_tasks():
    """Run once at startup (as root via systemd): refresh sudoers + tailscale serve."""
    import subprocess as _sp
    base = os.path.dirname(os.path.abspath(__file__))

    # Reinstall sudoers rules from the NAS mount (picks up any edits)
    sudoers_src = os.path.join(base, "prometheon-sudoers")
    sudoers_dst = "/etc/sudoers.d/prometheon"
    try:
        import shutil
        shutil.copy2(sudoers_src, sudoers_dst)
        os.chmod(sudoers_dst, 0o440)
        print("[startup] sudoers updated")
    except Exception as e:
        print(f"[startup] sudoers update skipped: {e}")

    # Allow zain to run tailscale without sudo in future
    try:
        _sp.run(["tailscale", "set", "--operator=zain"], timeout=10, capture_output=True)
        print("[startup] tailscale operator=zain set")
    except Exception as e:
        print(f"[startup] tailscale operator: {e}")

    # Set up tailscale serve so the app is reachable at https://prometheus
    try:
        _sp.run(["tailscale", "serve", "--bg", "http://localhost:8080"],
                timeout=15, capture_output=True)
        print("[startup] tailscale serve configured (https://prometheus)")
    except Exception as e:
        print(f"[startup] tailscale serve: {e}")

    # Provision/renew TLS certificate
    try:
        _sp.run(["tailscale", "cert", "prometheus.tail3045df.ts.net"],
                timeout=30, capture_output=True)
        print("[startup] TLS cert refreshed")
    except Exception as e:
        print(f"[startup] TLS cert: {e}")


if __name__ == "__main__":
    print("\n  ╔═══════════════════════════════════════╗")
    print("  ║       PROMETHEON NAS Terminal AI       ║")
    print("  ║       https://prometheus               ║")
    print("  ╚═══════════════════════════════════════╝\n")
    threading.Thread(target=_run_startup_tasks, daemon=True).start()
    import sys
    use_debug = "--no-debug" not in sys.argv
    extra_files = []
    if use_debug:
        for root, dirs, files in os.walk(os.path.join(_APP_DIR, "templates")):
            for f in files:
                extra_files.append(os.path.join(root, f))
        for root, dirs, files in os.walk(os.path.join(_APP_DIR, "static")):
            for f in files:
                if f.endswith((".css", ".js", ".svg")):
                    extra_files.append(os.path.join(root, f))
    app.run(
        host="0.0.0.0", port=8080, threaded=True,
        debug=use_debug, use_reloader=use_debug,
        extra_files=extra_files if use_debug else None,
    )
