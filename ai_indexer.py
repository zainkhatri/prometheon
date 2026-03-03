"""AI photo indexer for PROMETHEON.

Generates CLIP embeddings for semantic search ("dog", "beach at sunset",
"group selfie") and optionally detects + clusters faces for people browsing.

Uses thumbnails (already on disk) for speed — no need to read originals.
Processes incrementally: safe to re-run after adding new photos.

Prerequisites (install on the NAS):
    pip install torch open-clip-torch numpy pillow pillow-heif

For face detection (optional):
    sudo apt install cmake libboost-all-dev
    pip install face_recognition scikit-learn

Usage:
    python ai_indexer.py                # CLIP embeddings only
    python ai_indexer.py --faces        # Also detect + cluster faces
    python ai_indexer.py --recluster    # Re-cluster existing face data
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PHOTO_INDEX = SCRIPT_DIR / "photo_index.json"
AI_DIR = SCRIPT_DIR / "ai_data"
CLIP_HASHES_FILE = AI_DIR / "clip_hashes.json"
CLIP_EMB_FILE = AI_DIR / "clip_embeddings.npy"
FACE_INDEX_FILE = AI_DIR / "face_index.json"
FACE_EMB_FILE = AI_DIR / "face_embeddings.npy"
FACE_CLUSTERS_FILE = AI_DIR / "face_clusters.json"
FACE_CROPS_DIR = SCRIPT_DIR / "static" / "faces"
THUMB_DIR = SCRIPT_DIR / "static" / "thumbs"
SCREENSHOT_HASHES_FILE = AI_DIR / "screenshot_hashes.json"

SAVE_EVERY = 200


def thumb_hash(item):
    """Extract hash from thumbnail URL (unique key per photo)."""
    url = item.get("thumb", "")
    return url.rsplit("/", 1)[-1].replace(".jpg", "") if url else None


def load_image(item, max_size=512):
    """Load image for AI processing. Prefers thumbnail for speed."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    from PIL import Image

    h = thumb_hash(item)
    if h:
        tp = THUMB_DIR / f"{h}.jpg"
        if tp.exists():
            try:
                return Image.open(tp).convert("RGB")
            except Exception:
                pass

    path = item.get("path", "")
    if path and os.path.isfile(path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((max_size, max_size))
            return img
        except Exception:
            pass
    return None


# ─── CLIP Embeddings ────────────────────────────────────────────────

def load_clip_index():
    if not CLIP_HASHES_FILE.exists() or not CLIP_EMB_FILE.exists():
        return [], None
    try:
        with open(CLIP_HASHES_FILE) as f:
            hashes = json.load(f)
        emb = np.load(CLIP_EMB_FILE)
        return hashes, emb
    except Exception:
        return [], None


def save_clip_index(hashes, embeddings):
    AI_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLIP_HASHES_FILE, "w") as f:
        json.dump(hashes, f)
    np.save(CLIP_EMB_FILE, embeddings)


def scan_clip(photos):
    try:
        import torch
        import open_clip
    except ImportError:
        print("Missing deps. Install on the NAS:")
        print("  pip install torch open-clip-torch")
        sys.exit(1)

    old_hashes, old_emb = load_clip_index()
    old_set = set(old_hashes)
    todo = [p for p in photos if thumb_hash(p) and thumb_hash(p) not in old_set]

    if not todo:
        print(f"[clip] All {len(photos)} images already indexed.")
        return

    print(f"[clip] Loading CLIP ViT-B-32 model (first run downloads ~400 MB)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    print(f"[clip] Model ready. Processing {len(todo)} images ({len(old_hashes)} cached)...\n")

    new_h, new_e = [], []
    t0 = time.time()
    failed = 0

    for i, item in enumerate(todo):
        h = thumb_hash(item)
        img = load_image(item)
        if img is None:
            failed += 1
            continue

        try:
            tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                e = model.encode_image(tensor)
                e = (e / e.norm(dim=-1, keepdim=True)).squeeze().cpu().numpy()
            new_h.append(h)
            new_e.append(e)
        except Exception:
            failed += 1
            continue

        done = i + 1
        if done % 100 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(todo) - done) / rate
            print(f"  {done}/{len(todo)}  {rate:.1f} img/s  ETA {eta / 60:.0f}m")

        if done % SAVE_EVERY == 0:
            _merge_and_save(old_hashes, old_emb, new_h, new_e)

    _merge_and_save(old_hashes, old_emb, new_h, new_e)
    total = len(old_hashes) + len(new_h)
    elapsed = time.time() - t0
    print(f"\n[clip] Done: {len(new_h)} new, {failed} failed, {total} total  ({elapsed / 60:.1f}m)")


def _merge_and_save(old_h, old_e, new_h, new_e):
    if not new_h:
        return
    new_arr = np.array(new_e, dtype=np.float32)
    if old_e is not None and len(old_e):
        all_e = np.vstack([old_e, new_arr])
    else:
        all_e = new_arr
    save_clip_index(old_h + new_h, all_e)


# ─── Face Detection & Clustering ────────────────────────────────────

def scan_faces(photos, rescan=False):
    try:
        from insightface.app import FaceAnalysis
        import cv2
    except ImportError:
        print("[faces] Missing deps: pip install insightface onnxruntime opencv-python-headless")
        return

    FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    MIN_FACE_SIZE = 40  # px — skip tiny detections

    existing = {}
    existing_face_embs = []
    if not rescan:
        if FACE_INDEX_FILE.exists():
            with open(FACE_INDEX_FILE) as f:
                existing = json.load(f)
        if FACE_EMB_FILE.exists():
            existing_face_embs = list(np.load(FACE_EMB_FILE))

    todo = [p for p in photos if thumb_hash(p) and thumb_hash(p) not in existing]
    if not todo:
        print(f"[faces] All {len(photos)} images already scanned.")
        cluster_faces()
        return

    print("[faces] Loading InsightFace buffalo_l model…")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(960, 960))

    print(f"[faces] Scanning {len(todo)} images")
    print(f"[faces] {len(existing)} cached, {len(todo)} to process\n")

    all_embs = list(existing_face_embs)
    face_data = dict(existing)
    t0 = time.time()
    total_faces = len(all_embs)

    for i, item in enumerate(todo):
        h = thumb_hash(item)
        img_pil = load_image(item, max_size=1024)
        if img_pil is None:
            face_data[h] = []
            continue

        try:
            # InsightFace expects BGR numpy array
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            detected = app.get(img_bgr)

            # Filter tiny faces
            detected = [f for f in detected
                        if (f.bbox[2] - f.bbox[0]) >= MIN_FACE_SIZE
                        and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_SIZE]

            if not detected:
                face_data[h] = []
                continue

            faces = []
            for face in detected:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                emb_idx = len(all_embs)
                all_embs.append(face.normed_embedding.astype(np.float32))
                # Store bbox as [top, right, bottom, left] to match existing format
                faces.append({"bbox": [y1, x2, y2, x1], "emb_idx": emb_idx,
                              "det_score": round(float(face.det_score), 3)})

                # Save face crop
                pad = int(max(x2 - x1, y2 - y1) * 0.35)
                w, h_img = img_pil.width, img_pil.height
                crop = img_pil.crop((
                    max(0, x1 - pad), max(0, y1 - pad),
                    min(w, x2 + pad), min(h_img, y2 + pad),
                ))
                crop = crop.resize((150, 150))
                crop.save(FACE_CROPS_DIR / f"{emb_idx}.jpg", "JPEG", quality=85)
                total_faces += 1

            face_data[h] = faces
        except Exception as e:
            face_data[h] = []

        done = i + 1
        if done % 100 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(todo) - done) / rate
            print(f"  {done}/{len(todo)}  {rate:.1f} img/s  {total_faces} faces  ETA {eta / 60:.0f}m")

        if done % SAVE_EVERY == 0:
            AI_DIR.mkdir(parents=True, exist_ok=True)
            with open(FACE_INDEX_FILE, "w") as f:
                json.dump(face_data, f)
            if all_embs:
                np.save(FACE_EMB_FILE, np.array(all_embs, dtype=np.float32))

    AI_DIR.mkdir(parents=True, exist_ok=True)
    with open(FACE_INDEX_FILE, "w") as f:
        json.dump(face_data, f)
    if all_embs:
        np.save(FACE_EMB_FILE, np.array(all_embs, dtype=np.float32))

    elapsed = time.time() - t0
    print(f"\n[faces] Done: {total_faces} faces in {elapsed / 60:.1f}m")
    cluster_faces()


MIN_DET_SCORE = 0.5  # Minimum detection confidence for centroid/exemplar computation


def _build_score_map(face_data):
    """Build emb_idx → det_score lookup from face_data."""
    score_map = {}
    for ph, faces in face_data.items():
        for f in faces:
            score_map[f["emb_idx"]] = f.get("det_score", 1.0)  # default 1.0 for legacy
    return score_map


def _pick_exemplars(embs_subset, k=5):
    """Select up to k diverse exemplar embeddings using greedy max-min diversity."""
    if len(embs_subset) <= k:
        return embs_subset
    centroid = embs_subset.mean(axis=0)
    dists = np.linalg.norm(embs_subset - centroid, axis=1)
    picked = [int(np.argmin(dists))]  # start with closest to centroid
    for _ in range(k - 1):
        min_dists = np.min(
            [np.linalg.norm(embs_subset - embs_subset[p], axis=1) for p in picked],
            axis=0,
        )
        min_dists[picked] = -1  # exclude already picked
        picked.append(int(np.argmax(min_dists)))
    return embs_subset[picked]


def cluster_faces():
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        print("[faces] Install scikit-learn for face clustering: pip install scikit-learn")
        return

    if not FACE_EMB_FILE.exists() or not FACE_INDEX_FILE.exists():
        print("[faces] No face data to cluster.")
        return

    embs = np.load(FACE_EMB_FILE)
    with open(FACE_INDEX_FILE) as f:
        face_data = json.load(f)
    score_map = _build_score_map(face_data)

    if len(embs) == 0:
        print("[faces] No face embeddings found.")
        return

    INITIAL_THRESH = 0.9  # Higher = merges more aggressively
    MIN_FACES = 3
    MIN_PHOTOS = 8

    # Load existing clusters — named+curated ones become locked anchors
    old_clusters = {}
    if FACE_CLUSTERS_FILE.exists():
        try:
            with open(FACE_CLUSTERS_FILE) as f:
                old_clusters = json.load(f)
        except Exception:
            pass

    # Locked clusters: ALL named clusters are preserved exactly as-is
    locked = {cid: c for cid, c in old_clusters.items()
              if c.get("name")}
    old_names = {}
    for oc in old_clusters.values():
        if oc.get("name"):
            for ph in oc.get("photo_hashes", []):
                old_names[ph] = oc["name"]

    # Build emb_to_hash first (needed for locked anchor logic)
    emb_to_hash = {}
    for photo_hash, faces in face_data.items():
        for face in faces:
            emb_to_hash[face["emb_idx"]] = photo_hash

    from collections import Counter

    # Determine which embedding indices belong to locked clusters
    locked_photo_hashes = set()
    for c in locked.values():
        excluded = set(c.get("excluded_hashes", []))
        for ph in c.get("photo_hashes", []):
            if ph not in excluded:
                locked_photo_hashes.add(ph)

    locked_emb_indices = set()
    for emb_idx, ph in emb_to_hash.items():
        if ph in locked_photo_hashes:
            locked_emb_indices.add(emb_idx)

    if locked:
        print(f"[faces] Anchoring {len(locked)} named+curated people "
              f"({len(locked_photo_hashes)} photos, {len(locked_emb_indices)} embeddings locked)")

    # Cluster only the unlocked embeddings
    unlocked_mask = np.array([i not in locked_emb_indices for i in range(len(embs))])
    unlocked_embs = embs[unlocked_mask]
    unlocked_orig_indices = np.where(unlocked_mask)[0]

    print(f"[faces] Clustering {int(unlocked_mask.sum())} unlocked faces "
          f"(agglom average-linkage, threshold={INITIAL_THRESH})...")

    if len(unlocked_embs) >= 2:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=INITIAL_THRESH,
            metric="euclidean",
            linkage="average",
        )
        unlocked_labels = clustering.fit_predict(unlocked_embs)
    else:
        unlocked_labels = np.zeros(len(unlocked_embs), dtype=int)

    label_counts = Counter(unlocked_labels)

    raw_clusters = []
    for label in sorted(set(unlocked_labels)):
        if label_counts[label] < MIN_FACES:
            continue
        local_indices = np.where(unlocked_labels == label)[0]
        orig_indices = unlocked_orig_indices[local_indices]
        photo_hashes = list({emb_to_hash[int(idx)] for idx in orig_indices
                             if int(idx) in emb_to_hash and emb_to_hash[int(idx)] not in locked_photo_hashes})
        if len(photo_hashes) < MIN_PHOTOS:
            continue
        raw_clusters.append({"indices": orig_indices, "photo_hashes": photo_hashes})

    n_merged = 0

    # Assign names from old labels (skip names already claimed by locked clusters)
    locked_names = {c["name"] for c in locked.values()}
    for rc in raw_clusters:
        name_counts = Counter()
        for ph in rc["photo_hashes"]:
            if ph in old_names and old_names[ph] not in locked_names:
                name_counts[old_names[ph]] += 1
        rc["name"] = name_counts.most_common(1)[0][0] if name_counts else ""

    # Merge unlocked clusters that share the same user-assigned name
    name_groups = {}
    for i, rc in enumerate(raw_clusters):
        if rc["name"]:
            name_groups.setdefault(rc["name"], []).append(i)

    n_name_merged = 0
    merged_indices = set()
    final_clusters = []
    for name, group in name_groups.items():
        if len(group) > 1:
            base = raw_clusters[group[0]]
            for gi in group[1:]:
                other = raw_clusters[gi]
                base["indices"] = np.concatenate([base["indices"], other["indices"]])
                base["photo_hashes"] = list(set(base["photo_hashes"]) | set(other["photo_hashes"]))
                merged_indices.add(gi)
                n_name_merged += 1

    for i, rc in enumerate(raw_clusters):
        if i not in merged_indices:
            final_clusters.append(rc)

    if n_name_merged:
        print(f"[faces] Phase 3: merged {n_name_merged} clusters by matching names")

    def _best_sample(indices, centroid):
        dists = np.linalg.norm(embs[indices] - centroid, axis=1)
        best_emb_idx = int(indices[int(np.argmin(dists))])
        return emb_to_hash.get(best_emb_idx, "")

    clusters = {}
    used_ids = set()

    # Add locked anchors first — preserve their ORIGINAL cluster IDs
    for orig_cid, oc in locked.items():
        excluded = set(oc.get("excluded_hashes", []))
        photo_hashes = [ph for ph in oc["photo_hashes"] if ph not in excluded]
        if not photo_hashes:
            continue
        photo_set = set(photo_hashes)
        all_emb_indices = np.array([idx for idx, ph in emb_to_hash.items()
                                    if ph in photo_set and idx < len(embs)])
        if len(all_emb_indices) == 0:
            continue

        # If stored emb_indices exist from a previous run, use them as the
        # authoritative set; otherwise refine from all faces in cluster photos.
        stored_indices = oc.get("emb_indices")
        if stored_indices:
            valid = [i for i in stored_indices if i < len(embs)]
            if valid:
                c_emb_indices = np.array(valid)
            else:
                c_emb_indices = all_emb_indices
        else:
            # Iterative refinement: compute rough centroid from all faces,
            # then keep only the closest face per photo to filter out bystanders.
            rough_centroid = embs[all_emb_indices].mean(axis=0)
            refined = []
            for ph in photo_hashes:
                ph_indices = [idx for idx in all_emb_indices if emb_to_hash.get(int(idx)) == ph]
                if len(ph_indices) == 1:
                    refined.append(ph_indices[0])
                elif len(ph_indices) > 1:
                    dists = [(idx, np.linalg.norm(embs[idx] - rough_centroid)) for idx in ph_indices]
                    refined.append(min(dists, key=lambda x: x[1])[0])
            c_emb_indices = np.array(refined) if refined else all_emb_indices

        # Quality-gate: only high-confidence faces influence centroid/exemplars
        hq_indices = np.array([i for i in c_emb_indices
                               if score_map.get(int(i), 1.0) >= MIN_DET_SCORE])
        if len(hq_indices) == 0:
            hq_indices = c_emb_indices  # fallback if all low-quality

        centroid = embs[hq_indices].mean(axis=0)
        exemplars = _pick_exemplars(embs[hq_indices])
        entry = {
            "name": oc["name"],
            "photo_count": len(photo_hashes),
            "face_count": int(len(c_emb_indices)),
            "sample_face": _best_sample(c_emb_indices, centroid),
            "photo_hashes": photo_hashes,
            "excluded_hashes": list(excluded),
            "emb_indices": [int(i) for i in c_emb_indices],
            "exemplars": exemplars.tolist(),
        }
        # Preserve user-set fields (avatar, seeds, hidden, etc.)
        for k in ("avatar_hash", "avatar_bbox", "seed_hashes", "hidden"):
            if k in oc:
                entry[k] = oc[k]
        clusters[str(orig_cid)] = entry
        used_ids.add(int(orig_cid))

    # Find next available ID for new clusters (avoid collisions with locked IDs)
    next_id = max(used_ids, default=-1) + 1

    # Add newly clustered people
    for rc in sorted(final_clusters, key=lambda c: len(c["photo_hashes"]), reverse=True):
        while next_id in used_ids:
            next_id += 1
        hq_idx = np.array([i for i in rc["indices"]
                           if score_map.get(int(i), 1.0) >= MIN_DET_SCORE])
        if len(hq_idx) == 0:
            hq_idx = rc["indices"]  # fallback
        centroid = embs[hq_idx].mean(axis=0)
        exemplars = _pick_exemplars(embs[hq_idx])
        clusters[str(next_id)] = {
            "name": rc["name"],
            "photo_count": len(rc["photo_hashes"]),
            "face_count": int(len(rc["indices"])),
            "sample_face": _best_sample(rc["indices"], centroid),
            "photo_hashes": rc["photo_hashes"],
            "emb_indices": [int(i) for i in rc["indices"]],
            "exemplars": exemplars.tolist(),
        }
        used_ids.add(next_id)
        next_id += 1

    with open(FACE_CLUSTERS_FILE, "w") as f:
        json.dump(clusters, f, indent=2)

    n_small = sum(1 for c in label_counts.values() if c < MIN_FACES)
    named = sum(1 for c in clusters.values() if c["name"])
    print(f"[faces] {len(clusters)} people ({len(locked)} locked anchors), "
          f"{n_small} tiny filtered, {n_merged}+{n_name_merged} merged, {named} named")


def expand_named_clusters():
    """Sweep all face embeddings and assign unmatched ones to the nearest named cluster."""
    if not FACE_EMB_FILE.exists() or not FACE_INDEX_FILE.exists() or not FACE_CLUSTERS_FILE.exists():
        print("[expand] Missing face data. Run --faces first.")
        return

    embs = np.load(FACE_EMB_FILE)
    with open(FACE_INDEX_FILE) as f:
        face_data = json.load(f)
    with open(FACE_CLUSTERS_FILE) as f:
        clusters = json.load(f)
    score_map = _build_score_map(face_data)

    named = {cid: c for cid, c in clusters.items() if c.get("name")}
    if not named:
        print("[expand] No named clusters found. Name people in the People panel first.")
        return

    emb_to_hash = {}
    for ph, faces in face_data.items():
        for face in faces:
            emb_to_hash[face["emb_idx"]] = ph

    # Compute weighted centroid: seeds count 3x, original checkpoint photos 2x, auto-expanded 1x
    checkpoint_hashes = {}
    checkpoint_path = FACE_CLUSTERS_FILE.parent / "face_clusters.checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                cp = json.load(f)
            for cc in cp.values():
                if cc.get("name"):
                    for ph in cc.get("photo_hashes", []):
                        checkpoint_hashes[ph] = cc["name"]
        except Exception:
            pass

    exemplar_cids = []
    exemplar_list = []  # list of (k, emb_dim) arrays — one per cluster
    for cid, c in named.items():
        excluded = set(c.get("excluded_hashes", []))
        ph_set = set(c.get("photo_hashes", [])) - excluded
        seed_set = set(c.get("seed_hashes", []))
        name = c.get("name", "")
        cp_set = {ph for ph in ph_set if checkpoint_hashes.get(ph) == name}

        # Prefer stored exemplars (backward compat: fall back to weighted centroid)
        stored_exemplars = c.get("exemplars")
        if stored_exemplars and len(stored_exemplars) > 0:
            exemplar_cids.append(cid)
            exemplar_list.append(np.array(stored_exemplars, dtype=np.float32))
            continue

        # Legacy path: compute weighted centroid, wrap as single exemplar
        weighted_embs = []
        for idx, ph in emb_to_hash.items():
            if ph in ph_set and idx < len(embs):
                if score_map.get(idx, 1.0) < MIN_DET_SCORE:
                    continue
                if ph in seed_set:
                    w = 3.0
                elif ph in cp_set:
                    w = 2.0
                else:
                    w = 1.0
                weighted_embs.append((embs[idx], w))
        if weighted_embs:
            exemplar_cids.append(cid)
            vecs = np.array([e for e, w in weighted_embs])
            weights = np.array([w for e, w in weighted_embs])
            centroid = np.average(vecs, axis=0, weights=weights)
            exemplar_list.append(centroid.reshape(1, -1))

    if not exemplar_list:
        print("[expand] Could not compute any exemplars.")
        return

    print(f"[expand] {len(exemplar_cids)} named people, sweeping {len(emb_to_hash)} face embeddings...")

    # Confident match with clear margin
    ASSIGN_THRESH = 1.05  # looser distance threshold (was 0.95)
    MARGIN = 0.02  # smaller margin = accept closer calls (was 0.04)

    new_for_cluster = {cid: set() for cid in exemplar_cids}
    skipped_ambiguous = 0

    # Per-cluster: track which hashes are already in THAT cluster (not globally)
    cluster_hashes = {cid: set(clusters[cid].get("photo_hashes", []))
                      for cid in exemplar_cids}

    for emb_idx, ph in emb_to_hash.items():
        if emb_idx >= len(embs):
            continue
        emb = embs[emb_idx]
        # Min-exemplar distance: for each cluster, distance = min dist to any exemplar
        dists = np.array([float(np.min(np.linalg.norm(ex - emb, axis=1)))
                          for ex in exemplar_list])
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        if best_dist >= ASSIGN_THRESH:
            continue
        sorted_d = np.sort(dists)
        if len(sorted_d) > 1 and (sorted_d[1] - best_dist) < MARGIN:
            skipped_ambiguous += 1
            continue
        best_cid = exemplar_cids[best_idx]
        # Allow photo in multiple profiles — only skip if already in THIS cluster
        if ph not in cluster_hashes[best_cid]:
            new_for_cluster[best_cid].add(ph)

    total_added = 0
    for cid, new_hashes in new_for_cluster.items():
        if not new_hashes:
            continue
        c = clusters[cid]
        excluded = set(c.get("excluded_hashes", []))
        new_hashes -= excluded
        existing = set(c.get("photo_hashes", []))
        added = new_hashes - existing
        if not added:
            continue
        c["photo_hashes"] = list(existing | added)
        c["photo_count"] = len(c["photo_hashes"])
        total_added += len(added)
        print(f"  {c['name']}: +{len(added)} photos → {c['photo_count']} total")

    with open(FACE_CLUSTERS_FILE, "w") as f:
        json.dump(clusters, f, indent=2)

    print(f"[expand] Done. +{total_added} photos assigned, {skipped_ambiguous} ambiguous skipped.")


# ─── Video Face Detection ────────────────────────────────────────────

def scan_video_faces(videos):
    """Detect faces in videos by extracting frames with ffmpeg + InsightFace.

    Extracts 6 evenly-spaced frames per video, runs InsightFace on each frame,
    deduplicates faces found in multiple frames (same person), and appends to
    the existing face_index / face_embeddings. Videos whose faces are matched
    to a person automatically appear in their People profile.
    """
    try:
        from insightface.app import FaceAnalysis
        import cv2
    except ImportError:
        print("[video-faces] Missing deps: pip install insightface onnxruntime opencv-python-headless")
        return

    import subprocess
    import tempfile
    from PIL import Image as _PILImage

    FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    MIN_FACE_SIZE = 40     # px — skip tiny detections
    DEDUP_THRESH = 0.35    # L2 distance — faces this close across frames = same person

    # Load existing face data (built by scan_faces on photos)
    existing = {}
    all_embs = []
    if FACE_INDEX_FILE.exists():
        with open(FACE_INDEX_FILE) as f:
            existing = json.load(f)
    if FACE_EMB_FILE.exists():
        all_embs = list(np.load(FACE_EMB_FILE))

    todo = [v for v in videos if thumb_hash(v) and thumb_hash(v) not in existing]
    if not todo:
        print(f"[video-faces] All {len(videos)} videos already scanned.")
        return

    print("[video-faces] Loading InsightFace buffalo_l model…")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(960, 960))
    print(f"[video-faces] {len(videos)} videos total, {len(todo)} to process\n")

    face_data = dict(existing)
    t0 = time.time()
    total_new_faces = 0
    videos_with_faces = 0
    failed = 0

    for i, item in enumerate(todo):
        h = thumb_hash(item)
        video_path = item.get("path", "")

        if not video_path or not os.path.isfile(video_path):
            face_data[h] = []
            failed += 1
            continue

        # Get duration via ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=15,
            )
            duration = float(result.stdout.strip())
        except Exception:
            face_data[h] = []
            failed += 1
            continue

        if duration < 1.0:
            face_data[h] = []
            continue

        # Sample 6 timestamps: 10%, 25%, 40%, 55%, 70%, 85% of duration
        timestamps = [min(duration * f, duration - 0.1)
                      for f in [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]]

        frame_faces = []  # {emb, bbox, det_score, img_pil}
        with tempfile.TemporaryDirectory() as tmpdir:
            for ts in timestamps:
                frame_path = os.path.join(tmpdir, f"f{ts:.3f}.jpg")
                try:
                    subprocess.run(
                        ["ffmpeg", "-ss", str(ts), "-i", video_path,
                         "-frames:v", "1", "-q:v", "2", "-y", frame_path],
                        capture_output=True, timeout=20,
                    )
                except Exception:
                    continue
                if not os.path.exists(frame_path):
                    continue
                try:
                    img_pil = _PILImage.open(frame_path).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    detected = app.get(img_bgr)
                    detected = [f for f in detected
                                if (f.bbox[2] - f.bbox[0]) >= MIN_FACE_SIZE
                                and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_SIZE]
                    for face in detected:
                        frame_faces.append({
                            "emb": face.normed_embedding.astype(np.float32),
                            "bbox": face.bbox,
                            "det_score": float(face.det_score),
                            "img_pil": img_pil,
                        })
                except Exception:
                    continue

        if not frame_faces:
            face_data[h] = []
            continue

        # Deduplicate: same person in multiple frames → keep highest-confidence instance
        frame_faces.sort(key=lambda x: -x["det_score"])
        unique_faces = []
        for ff in frame_faces:
            is_dup = any(
                float(np.linalg.norm(uf["emb"] - ff["emb"])) < DEDUP_THRESH
                for uf in unique_faces
            )
            if not is_dup:
                unique_faces.append(ff)

        faces = []
        for ff in unique_faces:
            x1, y1, x2, y2 = [int(v) for v in ff["bbox"]]
            emb_idx = len(all_embs)
            all_embs.append(ff["emb"])
            # bbox stored as [top, right, bottom, left] — matches photo format
            faces.append({"bbox": [y1, x2, y2, x1], "emb_idx": emb_idx,
                          "det_score": round(float(ff["det_score"]), 3)})

            # Save 150×150 face crop (same as photos)
            img_pil = ff["img_pil"]
            pad = int(max(x2 - x1, y2 - y1) * 0.35)
            w, h_img = img_pil.width, img_pil.height
            crop = img_pil.crop((
                max(0, x1 - pad), max(0, y1 - pad),
                min(w, x2 + pad), min(h_img, y2 + pad),
            ))
            crop.resize((150, 150)).save(FACE_CROPS_DIR / f"{emb_idx}.jpg", "JPEG", quality=85)
            total_new_faces += 1

        face_data[h] = faces
        if faces:
            videos_with_faces += 1

        done = i + 1
        if done % 10 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(todo) - done) / rate
            print(f"  {done}/{len(todo)}  {rate:.2f} vid/s  "
                  f"{total_new_faces} new faces  ETA {eta / 60:.0f}m")

        if done % SAVE_EVERY == 0:
            AI_DIR.mkdir(parents=True, exist_ok=True)
            with open(FACE_INDEX_FILE, "w") as f:
                json.dump(face_data, f)
            if all_embs:
                np.save(FACE_EMB_FILE, np.array(all_embs, dtype=np.float32))

    AI_DIR.mkdir(parents=True, exist_ok=True)
    with open(FACE_INDEX_FILE, "w") as f:
        json.dump(face_data, f)
    if all_embs:
        np.save(FACE_EMB_FILE, np.array(all_embs, dtype=np.float32))

    elapsed = time.time() - t0
    print(f"\n[video-faces] Done: {total_new_faces} faces in "
          f"{videos_with_faces}/{len(todo)} videos  ({elapsed / 60:.1f}m)  {failed} failed")
    print("[video-faces] Running expand to funnel video faces into existing profiles…")
    expand_named_clusters()


# ─── Main ────────────────────────────────────────────────────────────

def resync_clusters(max_iters=10):
    """Reassign photos to correct clusters by iterating until centroids stabilize.

    Runs multiple passes: each pass recomputes centroids from the current
    (cleaner) photo_hashes, then reassigns. Stops when nothing changes.
    Respects excluded_hashes so manually removed photos stay removed.
    """
    if not FACE_EMB_FILE.exists() or not FACE_INDEX_FILE.exists() or not FACE_CLUSTERS_FILE.exists():
        print("[resync] Missing face data. Run --faces first.")
        return

    embs = np.load(FACE_EMB_FILE)
    with open(FACE_INDEX_FILE) as f:
        face_data = json.load(f)
    with open(FACE_CLUSTERS_FILE) as f:
        clusters = json.load(f)
    score_map = _build_score_map(face_data)

    named = {cid: c for cid, c in clusters.items() if c.get("name")}
    if not named:
        print("[resync] No named clusters. Name people in the People panel first.")
        return

    # Build emb_idx → photo_hash (static, doesn't change between iters)
    emb_to_hash = {}
    for ph, faces in face_data.items():
        for face in faces:
            emb_to_hash[face["emb_idx"]] = ph

    total_moved = 0
    total_added = 0

    for iteration in range(1, max_iters + 1):
        # Recompute exemplars from current photo_hashes each iteration
        exemplar_map = {}  # cid → array of exemplar embeddings
        for cid, c in named.items():
            excluded = set(c.get("excluded_hashes", []))
            hset = set(h for h in c.get("photo_hashes", []) if h not in excluded)
            all_idxs = [idx for idx, ph in emb_to_hash.items() if ph in hset]
            # Quality-gate: only high-confidence faces
            hq_idxs = [i for i in all_idxs if score_map.get(i, 1.0) >= MIN_DET_SCORE]
            if not hq_idxs:
                hq_idxs = all_idxs  # fallback
            if hq_idxs:
                exemplar_map[cid] = _pick_exemplars(embs[hq_idxs])

        if not exemplar_map:
            print("[resync] Could not compute any exemplars.")
            return

        cids = list(exemplar_map.keys())
        exemplar_arrays = [exemplar_map[c] for c in cids]

        # Assign each embedding to nearest cluster via min-exemplar distance.
        # If two clusters are within MARGIN of each other, the embedding is
        # considered ambiguous and kept in its current cluster (no forced move).
        MARGIN = 0.08
        emb_to_cluster = {}  # only confident assignments
        for emb_idx in emb_to_hash:
            if emb_idx < len(embs):
                emb = embs[emb_idx]
                dists = np.array([float(np.min(np.linalg.norm(ex - emb, axis=1)))
                                  for ex in exemplar_arrays])
                order = np.argsort(dists)
                best, second = dists[order[0]], dists[order[1]]
                if second - best >= MARGIN:  # confident enough
                    emb_to_cluster[emb_idx] = cids[int(order[0])]

        # Map photo → set of clusters its faces confidently belong to
        photo_to_clusters = {}
        for emb_idx, cid in emb_to_cluster.items():
            ph = emb_to_hash.get(emb_idx)
            if ph:
                photo_to_clusters.setdefault(ph, set()).add(cid)

        moved = added = 0
        for cid, c in named.items():
            excluded = set(c.get("excluded_hashes", []))
            current_hashes = set(c.get("photo_hashes", []))
            new_hashes = set()

            for ph in current_hashes:
                if ph in excluded:
                    continue
                confident = photo_to_clusters.get(ph)
                if confident is None:
                    # No confident assignment for any face — leave it in place
                    new_hashes.add(ph)
                elif cid in confident:
                    new_hashes.add(ph)  # correctly placed
                else:
                    moved += 1  # all confident faces point elsewhere

            # Add photos that should be in this cluster (once per cluster, not per photo)
            for ph, should_be_in in photo_to_clusters.items():
                if cid in should_be_in and ph not in excluded and ph not in new_hashes:
                    new_hashes.add(ph)
                    if ph not in current_hashes:
                        added += 1

            c["photo_hashes"] = sorted(new_hashes)
            c["photo_count"] = len(new_hashes)

        total_moved += moved
        total_added += added
        print(f"  iter {iteration}: {moved} removed, {added} added")
        if moved == 0 and added == 0:
            print(f"[resync] Converged after {iteration} iteration(s).")
            break
    else:
        print(f"[resync] Reached max iterations ({max_iters}), may not be fully converged.")

    with open(FACE_CLUSTERS_FILE, "w") as f:
        json.dump(clusters, f, indent=2)

    print(f"\n[resync] Total: {total_moved} removed, {total_added} added.")
    for cid, c in named.items():
        print(f"  {c['name']}: {c['photo_count']} photos")



# ─── Screenshot / Document Classification ────────────────────────────

SCREENSHOT_QUERIES = [
    "screenshot of phone screen",
    "text messages conversation screenshot",
    "scanned document or paper",
    "receipt or invoice",
    "computer screen screenshot",
    "text on a screen or display",
    "screenshot of a website or app",
    "meme with text overlay on image",
]

REAL_PHOTO_QUERIES = [
    "a photograph of people",
    "outdoor nature landscape photograph",
    "a real camera photo of a scene",
    "portrait photograph",
]

# Screenshots that contain people should stay in the gallery
PEOPLE_QUERIES = [
    "photo of a person or people",
    "selfie or group photo",
    "someone's face",
]

PEOPLE_THRESH = 0.23  # if screenshot scores above this for people, keep it

# High-confidence: screenshot score alone is enough
CLIP_HIGH_THRESH = 0.27
# Differential: screenshot_score - real_score > 0 means more screenshot-like
CLIP_DIFF_THRESH = 0.0


def classify_screenshots(photos):
    """Classify photos as screenshots/documents using CLIP differential scoring + heuristics."""
    try:
        import torch
        import open_clip
    except ImportError:
        print("Missing deps. Install on the NAS:")
        print("  pip install torch open-clip-torch")
        sys.exit(1)

    clip_hashes, clip_emb = load_clip_index()
    if clip_emb is None or len(clip_hashes) == 0:
        print("[screenshots] No CLIP embeddings found. Run ai_indexer.py first.")
        return

    # Build hash→photo item for filename lookups
    hash_to_item = {}
    for p in photos:
        h = thumb_hash(p)
        if h:
            hash_to_item[h] = p

    # Load face index to exclude photos with detected faces
    faces_with_people = set()
    if FACE_INDEX_FILE.exists():
        with open(FACE_INDEX_FILE) as f:
            face_data = json.load(f)
        for ph, faces in face_data.items():
            if faces:
                faces_with_people.add(ph)
    print(f"[screenshots] {len(faces_with_people)} photos have detected faces (will be excluded)")

    # ── CLIP-based scoring ──
    print("[screenshots] Loading CLIP model for text encoding…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    print("[screenshots] Encoding text queries…")
    with torch.no_grad():
        ss_tokens = tokenizer(SCREENSHOT_QUERIES)
        ss_emb = model.encode_text(ss_tokens)
        ss_emb = (ss_emb / ss_emb.norm(dim=-1, keepdim=True)).cpu().numpy()

        real_tokens = tokenizer(REAL_PHOTO_QUERIES)
        real_emb = model.encode_text(real_tokens)
        real_emb = (real_emb / real_emb.norm(dim=-1, keepdim=True)).cpu().numpy()

        people_tokens = tokenizer(PEOPLE_QUERIES)
        people_emb = model.encode_text(people_tokens)
        people_emb = (people_emb / people_emb.norm(dim=-1, keepdim=True)).cpu().numpy()

    # Score all images
    ss_scores = (clip_emb @ ss_emb.T).max(axis=1)
    real_scores = (clip_emb @ real_emb.T).max(axis=1)
    people_scores = (clip_emb @ people_emb.T).max(axis=1)
    diff_scores = ss_scores - real_scores

    hash_to_idx = {h: i for i, h in enumerate(clip_hashes)}

    # High-confidence CLIP: screenshot score alone is high enough
    clip_flagged = set()
    for i, h in enumerate(clip_hashes):
        if ss_scores[i] >= CLIP_HIGH_THRESH:
            clip_flagged.add(h)
    print(f"[screenshots] CLIP high-confidence: {len(clip_flagged)} (ss_score >= {CLIP_HIGH_THRESH})")

    # Differential CLIP for PNGs: screenshot score beats real photo score
    # Only apply to PNGs — JPGs/HEICs are camera photos where high-conf is enough
    png_hashes = set()
    for p in photos:
        h = thumb_hash(p)
        if h and p.get("path", "").lower().endswith(".png"):
            png_hashes.add(h)

    diff_flagged = set()
    for i, h in enumerate(clip_hashes):
        if h in png_hashes and diff_scores[i] > CLIP_DIFF_THRESH and h not in clip_flagged:
            diff_flagged.add(h)
    print(f"[screenshots] CLIP differential (PNGs only): +{len(diff_flagged)} (ss > real)")

    # ── Filename heuristics (covers photos without CLIP embeddings) ──
    name_flagged = set()
    clip_hash_set = set(clip_hashes)
    for p in photos:
        h = thumb_hash(p)
        if not h:
            continue
        path = p.get("path", "").lower()
        fname = os.path.basename(path)
        # PNGs without CLIP embeddings — can't score them, assume screenshot
        if path.endswith(".png") and h not in clip_hash_set:
            name_flagged.add(h)
        # Files with screenshot-related names
        for kw in ("screenshot", "screen shot", "screen_shot", "screencapture", "screen capture"):
            if kw in fname:
                name_flagged.add(h)
                break

    print(f"[screenshots] Filename heuristics: +{len(name_flagged)}")

    # ── Combine and exclude photos with faces or people ──
    all_flagged = clip_flagged | diff_flagged | name_flagged
    before_exclude = len(all_flagged)
    all_flagged -= faces_with_people
    face_excluded = before_exclude - len(all_flagged)
    print(f"[screenshots] {face_excluded} excluded (have detected faces)")

    # Also exclude screenshots that CLIP thinks contain people
    people_excluded = set()
    for h in list(all_flagged):
        idx = hash_to_idx.get(h)
        if idx is not None and people_scores[idx] >= PEOPLE_THRESH:
            people_excluded.add(h)
    all_flagged -= people_excluded
    print(f"[screenshots] {len(people_excluded)} excluded (CLIP detected people in screenshot)")
    print(f"[screenshots] Final: {len(all_flagged)} screenshots/documents")

    # Save
    AI_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCREENSHOT_HASHES_FILE, "w") as f:
        json.dump(sorted(all_flagged), f)

    print(f"[screenshots] Saved to {SCREENSHOT_HASHES_FILE}")


def main():
    parser = argparse.ArgumentParser(description="AI photo indexer for PROMETHEON")
    parser.add_argument("--faces", action="store_true", help="Also detect and cluster faces")
    parser.add_argument("--recluster", action="store_true", help="Re-cluster faces without re-scanning")
    parser.add_argument("--expand", action="store_true", help="Assign unmatched faces to nearest named person")
    parser.add_argument("--resync", action="store_true", help="Reassign photos to correct clusters based on face centroids")
    parser.add_argument("--rescan-faces", action="store_true", help="Wipe face data and rescan from scratch (higher quality)")
    parser.add_argument("--video-faces", action="store_true", help="Detect faces in videos using ffmpeg frame extraction")
    parser.add_argument("--classify-screenshots", action="store_true", help="Classify screenshots/documents via CLIP and remove from gallery")
    args = parser.parse_args()

    if not PHOTO_INDEX.exists():
        print(f"Photo index not found: {PHOTO_INDEX}")
        print("Run photo_scanner.py first.")
        sys.exit(1)

    with open(PHOTO_INDEX) as f:
        photos = json.load(f)

    images = [p for p in photos if p.get("type") != "video"]
    videos = [p for p in photos if p.get("type") == "video"]
    print(f"Photo index: {len(images)} images, {len(videos)} videos\n")

    if args.recluster:
        cluster_faces()
        return

    if args.expand:
        expand_named_clusters()
        return

    if args.resync:
        resync_clusters()
        return

    if args.rescan_faces:
        print("[faces] Wiping old face data for fresh high-quality rescan...")
        for f in [FACE_INDEX_FILE, FACE_EMB_FILE, FACE_CLUSTERS_FILE]:
            if f.exists():
                f.unlink()
        scan_faces(images, rescan=True)
        print("\nRestart the PROMETHEON server to enable AI search.")
        return

    if args.video_faces:
        scan_video_faces(videos)
        print("\nRestart the PROMETHEON server to reload face clusters.")
        return

    if args.classify_screenshots:
        classify_screenshots(images)
        print("\nRestart the PROMETHEON server to apply screenshot filtering.")
        return

    scan_clip(images)

    if args.faces:
        print()
        scan_faces(images)

    print("\nRestart the PROMETHEON server to enable AI search.")


if __name__ == "__main__":
    main()
