# MeetingGenie - Speaker Signature Database
# Stores voice embeddings per identified speaker so they can be
# recognized automatically across future transcriptions.

import os
import json
from datetime import date

import appdirs

# Cosine-similarity threshold above which we consider two embeddings
# to belong to the same person.
SIMILARITY_THRESHOLD = 0.75


def _db_path() -> str:
    config_dir = appdirs.user_config_dir('MeetingGenie')
    return os.path.join(config_dir, 'speaker_signatures.json')


def load_db() -> dict:
    """Return the full database dict, or an empty one if not found / corrupt."""
    path = _db_path()
    if not os.path.exists(path):
        return {"speakers": []}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"speakers": []}


def save_db(db: dict) -> None:
    """Write the database dict to disk."""
    path = _db_path()
    db_dir = os.path.dirname(path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def _cosine_similarity(e1: list, e2: list) -> float:
    """Return cosine similarity between two embedding lists."""
    try:
        import numpy as np
        a = np.array(e1, dtype=np.float32)
        b = np.array(e2, dtype=np.float32)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a / na, b / nb))
    except Exception:
        return 0.0


def find_match(embedding: list, threshold: float = SIMILARITY_THRESHOLD):
    """
    Compare *embedding* against every stored speaker.

    Returns (name, similarity) where *name* is the best-matching stored
    speaker name and *similarity* is the cosine similarity score.
    If no stored speaker reaches *threshold*, name will be None.
    """
    db = load_db()
    best_name = None
    best_sim = 0.0
    for speaker in db.get("speakers", []):
        stored = speaker.get("embedding")
        if not stored:
            continue
        sim = _cosine_similarity(embedding, stored)
        if sim > best_sim:
            best_sim = sim
            best_name = speaker["name"]
    if best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim


def save_speaker(name: str, embedding: list) -> None:
    """
    Add or update a speaker entry.

    If a speaker with the same name (case-insensitive) already exists, the
    stored embedding is blended with the new one so the model gradually
    adapts to variations over time.
    """
    try:
        import numpy as np
    except ImportError:
        return

    db = load_db()
    today = str(date.today())
    name = name.strip()

    # Normalise the incoming embedding
    emb = np.array(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm < 1e-6:
        return
    emb = emb / norm

    # Update existing entry if the name matches
    for speaker in db.get("speakers", []):
        if speaker["name"].strip().lower() == name.lower():
            existing = np.array(speaker["embedding"], dtype=np.float32)
            blended = (existing + emb) / 2.0
            b_norm = float(np.linalg.norm(blended))
            if b_norm > 1e-6:
                blended = blended / b_norm
            speaker["embedding"] = blended.tolist()
            speaker["updated"] = today
            save_db(db)
            return

    # New speaker
    db.setdefault("speakers", []).append({
        "name": name,
        "embedding": emb.tolist(),
        "created": today,
        "updated": today,
    })
    save_db(db)


def list_speakers() -> list:
    """Return the list of stored speaker names."""
    db = load_db()
    return [s["name"] for s in db.get("speakers", [])]


def delete_speaker(name: str) -> None:
    """Remove all entries whose name matches (case-insensitive)."""
    db = load_db()
    db["speakers"] = [
        s for s in db.get("speakers", [])
        if s["name"].strip().lower() != name.strip().lower()
    ]
    save_db(db)


def rename_speaker(old_name: str, new_name: str) -> bool:
    """Rename a speaker. Returns True if found and renamed."""
    db = load_db()
    found = False
    for s in db.get("speakers", []):
        if s["name"].strip().lower() == old_name.strip().lower():
            s["name"] = new_name.strip()
            found = True
    if found:
        save_db(db)
    return found


def merge_speakers(name_keep: str, name_absorb: str) -> bool:
    """Merge two speakers: blend embeddings and keep name_keep. Deletes name_absorb."""
    try:
        import numpy as np
    except ImportError:
        return False

    db = load_db()
    speakers = db.get("speakers", [])
    keep = None
    absorb = None
    for s in speakers:
        if s["name"].strip().lower() == name_keep.strip().lower():
            keep = s
        elif s["name"].strip().lower() == name_absorb.strip().lower():
            absorb = s

    if not keep or not absorb:
        return False

    e1 = np.array(keep["embedding"], dtype=np.float32)
    e2 = np.array(absorb["embedding"], dtype=np.float32)
    blended = (e1 + e2) / 2.0
    norm = float(np.linalg.norm(blended))
    if norm > 1e-6:
        blended = blended / norm
    keep["embedding"] = blended.tolist()
    keep["updated"] = str(date.today())

    db["speakers"] = [s for s in speakers if s is not absorb]
    save_db(db)
    return True


def get_speaker_info(name: str) -> dict:
    """Get full speaker info dict (name, created, updated). None if not found."""
    db = load_db()
    for s in db.get("speakers", []):
        if s["name"].strip().lower() == name.strip().lower():
            return {
                "name": s["name"],
                "created": s.get("created", "?"),
                "updated": s.get("updated", "?"),
                "embedding_size": len(s.get("embedding", [])),
            }
    return None


def get_similarity(name1: str, name2: str) -> float:
    """Get cosine similarity between two stored speakers."""
    db = load_db()
    e1 = e2 = None
    for s in db.get("speakers", []):
        if s["name"].strip().lower() == name1.strip().lower():
            e1 = s.get("embedding")
        elif s["name"].strip().lower() == name2.strip().lower():
            e2 = s.get("embedding")
    if e1 and e2:
        return _cosine_similarity(e1, e2)
    return 0.0
