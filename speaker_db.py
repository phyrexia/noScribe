# MeetingGenie - Speaker Signature Database
# Stores voice embeddings + audio samples per identified speaker so they can
# be recognized automatically across future transcriptions and managed by the
# user from the Settings page.

import os
import json
import uuid as _uuid
from datetime import datetime, date

import appdirs

# Cosine-similarity threshold above which we consider two embeddings to belong
# to the same person.
SIMILARITY_THRESHOLD = 0.75

# Maximum number of WAV samples kept per signature (oldest dropped first).
MAX_SAMPLES = 4

# Maximum number of raw embeddings remembered for re-averaging.
MAX_EMBEDDINGS_HISTORY = 5


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────


def _config_dir() -> str:
    return appdirs.user_config_dir('MeetingGenie')


def _db_path() -> str:
    return os.path.join(_config_dir(), 'speaker_signatures.json')


def _samples_dir() -> str:
    """Absolute path of the directory where WAV samples are stored."""
    p = os.path.join(_config_dir(), 'voice_samples')
    if not os.path.exists(p):
        try:
            os.makedirs(p)
        except OSError:
            pass
    return p


def sample_abs_path(rel_or_abs: str) -> str:
    """Resolve a sample path stored in the DB to an absolute path on disk."""
    if not rel_or_abs:
        return rel_or_abs
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(_config_dir(), rel_or_abs)


# ─────────────────────────────────────────────────────────────────────────────
# Load / save with on-the-fly migration
# ─────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec='seconds') + 'Z'


def _migrate_entry(entry: dict, today: str) -> bool:
    """Backfill new schema fields on a legacy entry. Returns True if changed."""
    changed = False
    if 'id' not in entry or not entry['id']:
        entry['id'] = str(_uuid.uuid4())
        changed = True
    if 'samples' not in entry or entry.get('samples') is None:
        entry['samples'] = []
        changed = True
    if 'embeddings_history' not in entry:
        # Seed history with current embedding so future re-averaging has data.
        emb = entry.get('embedding')
        entry['embeddings_history'] = [list(emb)] if emb else []
        changed = True
    if 'use_count' not in entry:
        entry['use_count'] = 1
        changed = True
    if 'last_used' not in entry:
        entry['last_used'] = entry.get('updated') or entry.get('created') or today
        changed = True
    if 'source_file' not in entry:
        entry['source_file'] = ''
        changed = True
    if 'created' not in entry:
        entry['created'] = today
        changed = True
    if 'updated' not in entry:
        entry['updated'] = today
        changed = True
    return changed


def load_db() -> dict:
    """Return the full database dict, migrating legacy entries in place.

    On-disk migration is persisted automatically (best effort).
    """
    path = _db_path()
    if not os.path.exists(path):
        return {"speakers": []}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            db = json.load(f)
    except Exception:
        return {"speakers": []}

    today = str(date.today())
    changed = False
    for s in db.get('speakers', []):
        if _migrate_entry(s, today):
            changed = True

    if changed:
        try:
            save_db(db)
        except Exception:
            pass
    return db


def save_db(db: dict) -> None:
    """Write the database dict to disk."""
    path = _db_path()
    db_dir = os.path.dirname(path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_similarity(e1, e2) -> float:
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


def _normalise(emb):
    import numpy as np
    a = np.array(emb, dtype=np.float32)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        return None
    return a / n


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────


def find_match(embedding, threshold: float = SIMILARITY_THRESHOLD):
    """Compare *embedding* against every stored speaker.

    Returns (name, uuid, similarity). When no stored speaker reaches
    *threshold*, name and uuid are None but the best score is still returned
    (callers can decide what to do with sub-threshold matches).
    """
    db = load_db()
    best_name = None
    best_uuid = None
    best_sim = 0.0
    for speaker in db.get("speakers", []):
        stored = speaker.get("embedding")
        if not stored:
            continue
        sim = _cosine_similarity(embedding, stored)
        if sim > best_sim:
            best_sim = sim
            best_name = speaker["name"]
            best_uuid = speaker.get("id")
    if best_sim >= threshold:
        return best_name, best_uuid, best_sim
    return None, None, best_sim


# ─────────────────────────────────────────────────────────────────────────────
# Mutations
# ─────────────────────────────────────────────────────────────────────────────


def add_speaker(
    name: str,
    embedding,
    samples=None,
    source_file: str = '',
    desired_uuid: str = '',
) -> str:
    """Add a brand-new speaker entry and return its UUID.

    If a speaker with the same name already exists, the embedding is blended
    into the existing entry (use_count incremented) and the existing UUID is
    returned. Provide *samples* as a list of paths (relative to the config
    dir, or absolute — stored as relative when inside it). When *desired_uuid*
    is provided and unused, the new entry is created with that UUID — handy
    for keeping pre-generated sample filenames in sync with the DB.
    """
    import numpy as np

    name = (name or '').strip()
    if not name:
        return ''

    emb_norm = _normalise(embedding)
    if emb_norm is None:
        return ''

    db = load_db()
    now = _now_iso()
    today = str(date.today())
    samples = list(samples or [])
    # Normalise sample paths to be relative to the config dir when possible.
    cdir = _config_dir()
    norm_samples = []
    for sp in samples:
        if not sp:
            continue
        if os.path.isabs(sp):
            try:
                rel = os.path.relpath(sp, cdir)
                if not rel.startswith('..'):
                    sp = rel
            except ValueError:
                pass
        norm_samples.append(sp)

    # If a speaker with the same name exists, update it instead of duplicating.
    for s in db.get('speakers', []):
        if s['name'].strip().lower() == name.lower():
            existing = np.array(s['embedding'], dtype=np.float32)
            blended = (existing + emb_norm) / 2.0
            bnorm = float(np.linalg.norm(blended))
            if bnorm > 1e-6:
                blended = blended / bnorm
            s['embedding'] = blended.tolist()
            history = s.setdefault('embeddings_history', [])
            history.append(emb_norm.tolist())
            del history[:-MAX_EMBEDDINGS_HISTORY]
            # Append new samples, capped at MAX_SAMPLES (oldest first dropped).
            existing_samples = list(s.get('samples', []))
            existing_samples.extend(norm_samples)
            if len(existing_samples) > MAX_SAMPLES:
                # Remove sample files we're dropping from disk.
                for dropped in existing_samples[:-MAX_SAMPLES]:
                    try:
                        os.remove(sample_abs_path(dropped))
                    except OSError:
                        pass
                existing_samples = existing_samples[-MAX_SAMPLES:]
            s['samples'] = existing_samples
            s['use_count'] = int(s.get('use_count', 1)) + 1
            s['updated'] = today
            s['last_used'] = now
            if source_file:
                s['source_file'] = source_file
            save_db(db)
            return s['id']

    # Otherwise create a new entry. Honour desired_uuid when collision-free.
    used = {s.get('id') for s in db.get('speakers', [])}
    if desired_uuid and desired_uuid not in used:
        new_id = desired_uuid
    else:
        new_id = str(_uuid.uuid4())
    db.setdefault('speakers', []).append({
        'id': new_id,
        'name': name,
        'embedding': emb_norm.tolist(),
        'embeddings_history': [emb_norm.tolist()],
        'samples': norm_samples,
        'created': today,
        'updated': today,
        'last_used': now,
        'use_count': 1,
        'source_file': source_file or '',
    })
    save_db(db)
    return new_id


def save_speaker(
    name: str,
    embedding,
    samples=None,
    source_file: str = '',
    desired_uuid: str = '',
) -> str:
    """Backward-compatible wrapper used by the speaker-naming dialog.

    Returns the speaker UUID (new or existing).
    """
    return add_speaker(
        name,
        embedding,
        samples=samples,
        source_file=source_file,
        desired_uuid=desired_uuid,
    )


def update_speaker_last_used(uuid: str) -> None:
    """Bump *last_used* and increment *use_count* for the given speaker."""
    if not uuid:
        return
    db = load_db()
    for s in db.get('speakers', []):
        if s.get('id') == uuid:
            s['last_used'] = _now_iso()
            s['use_count'] = int(s.get('use_count', 0)) + 1
            save_db(db)
            return


def attach_samples(uuid: str, sample_paths) -> None:
    """Attach freshly-captured sample WAVs to an existing speaker."""
    if not uuid or not sample_paths:
        return
    cdir = _config_dir()
    rel_paths = []
    for sp in sample_paths:
        if not sp:
            continue
        if os.path.isabs(sp):
            try:
                rel = os.path.relpath(sp, cdir)
                if not rel.startswith('..'):
                    sp = rel
            except ValueError:
                pass
        rel_paths.append(sp)

    db = load_db()
    for s in db.get('speakers', []):
        if s.get('id') == uuid:
            existing = list(s.get('samples', []))
            existing.extend(rel_paths)
            if len(existing) > MAX_SAMPLES:
                for dropped in existing[:-MAX_SAMPLES]:
                    try:
                        os.remove(sample_abs_path(dropped))
                    except OSError:
                        pass
                existing = existing[-MAX_SAMPLES:]
            s['samples'] = existing
            s['updated'] = str(date.today())
            save_db(db)
            return


def list_speakers() -> list:
    """Return the list of stored speaker names (legacy API)."""
    db = load_db()
    return [s["name"] for s in db.get("speakers", [])]


def list_speakers_full() -> list:
    """Return full speaker dicts for UI rendering."""
    return load_db().get("speakers", [])


def get_speaker(uuid: str) -> dict:
    db = load_db()
    for s in db.get('speakers', []):
        if s.get('id') == uuid:
            return s
    return None


def delete_speaker(uuid_or_name: str) -> bool:
    """Remove a speaker by UUID or name. Cleans up sample WAVs on disk."""
    if not uuid_or_name:
        return False
    db = load_db()
    target = (uuid_or_name or '').strip().lower()
    kept = []
    removed_any = False
    for s in db.get('speakers', []):
        sid = (s.get('id') or '').lower()
        sname = (s.get('name') or '').strip().lower()
        if sid == target or sname == target:
            for sp in s.get('samples', []) or []:
                try:
                    os.remove(sample_abs_path(sp))
                except OSError:
                    pass
            removed_any = True
            continue
        kept.append(s)
    if removed_any:
        db['speakers'] = kept
        save_db(db)
    return removed_any


def rename_speaker(old: str, new_name: str) -> bool:
    """Rename a speaker. Lookup by UUID first, then name. Returns success."""
    if not old or not new_name:
        return False
    new_name = new_name.strip()
    if not new_name:
        return False
    db = load_db()
    found = False
    target = old.strip().lower()
    for s in db.get('speakers', []):
        sid = (s.get('id') or '').lower()
        sname = (s.get('name') or '').strip().lower()
        if sid == target or sname == target:
            s['name'] = new_name
            s['updated'] = str(date.today())
            found = True
    if found:
        save_db(db)
    return found


def merge_speakers(uuid_a: str, uuid_b: str) -> str:
    """Merge B into A. Returns the surviving UUID (A), or '' on failure.

    - Embedding: weighted average by ``use_count``.
    - Samples: concatenated, oldest dropped if over ``MAX_SAMPLES``.
    - ``embeddings_history`` combined (capped).
    - ``use_count`` summed.
    - ``last_used`` = max.
    - B's sample files removed from disk.

    *uuid_a* / *uuid_b* may also be names for backward compatibility, though
    UUIDs are preferred.
    """
    import numpy as np

    if not uuid_a or not uuid_b or uuid_a == uuid_b:
        return ''

    db = load_db()
    speakers = db.get('speakers', [])

    def _find(key):
        kl = key.strip().lower()
        for s in speakers:
            if (s.get('id') or '').lower() == kl:
                return s
        for s in speakers:
            if (s.get('name') or '').strip().lower() == kl:
                return s
        return None

    a = _find(uuid_a)
    b = _find(uuid_b)
    if a is None or b is None or a is b:
        return ''

    wa = max(int(a.get('use_count', 1)), 1)
    wb = max(int(b.get('use_count', 1)), 1)
    ea = np.array(a['embedding'], dtype=np.float32)
    eb = np.array(b['embedding'], dtype=np.float32)
    merged = (ea * wa + eb * wb) / float(wa + wb)
    n = float(np.linalg.norm(merged))
    if n > 1e-6:
        merged = merged / n
    a['embedding'] = merged.tolist()

    # Combine embedding histories
    history = list(a.get('embeddings_history', []))
    history.extend(b.get('embeddings_history', []) or [])
    a['embeddings_history'] = history[-MAX_EMBEDDINGS_HISTORY:]

    # Combine samples (a first, then b). Drop b's overflow files from disk.
    combined_samples = list(a.get('samples', []) or [])
    combined_samples.extend(b.get('samples', []) or [])
    if len(combined_samples) > MAX_SAMPLES:
        for dropped in combined_samples[:-MAX_SAMPLES]:
            try:
                os.remove(sample_abs_path(dropped))
            except OSError:
                pass
        combined_samples = combined_samples[-MAX_SAMPLES:]
    a['samples'] = combined_samples

    a['use_count'] = wa + wb

    # last_used = max
    la = a.get('last_used') or ''
    lb = b.get('last_used') or ''
    a['last_used'] = max(la, lb)
    a['updated'] = str(date.today())

    # Remove any of B's sample files that did NOT survive in A (already removed
    # above). B's files that DID survive are now owned by A.
    kept_set = set(a['samples'])
    for sp in b.get('samples', []) or []:
        if sp not in kept_set:
            try:
                os.remove(sample_abs_path(sp))
            except OSError:
                pass

    db['speakers'] = [s for s in speakers if s is not b]
    save_db(db)
    return a.get('id', '')


def pairwise_similarity():
    """Return a list of ``(uuid_a, uuid_b, cosine)`` for every speaker pair.

    Sorted by similarity descending. Names/uuids are deterministic per pair
    (lexicographically smaller id first), so the same pair won't appear twice.
    """
    db = load_db()
    speakers = db.get('speakers', [])
    out = []
    for i in range(len(speakers)):
        si = speakers[i]
        ei = si.get('embedding')
        if not ei:
            continue
        for j in range(i + 1, len(speakers)):
            sj = speakers[j]
            ej = sj.get('embedding')
            if not ej:
                continue
            sim = _cosine_similarity(ei, ej)
            ida = si.get('id') or si.get('name', '')
            idb = sj.get('id') or sj.get('name', '')
            if ida > idb:
                ida, idb = idb, ida
            out.append((ida, idb, sim))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def get_speaker_info(name_or_uuid: str) -> dict:
    """Get full speaker info dict, looked up by UUID or name."""
    db = load_db()
    target = (name_or_uuid or '').strip().lower()
    for s in db.get("speakers", []):
        if (s.get('id') or '').lower() == target or s.get('name', '').strip().lower() == target:
            return {
                'id': s.get('id'),
                "name": s["name"],
                "created": s.get("created", "?"),
                "updated": s.get("updated", "?"),
                "last_used": s.get("last_used", "?"),
                "use_count": s.get("use_count", 0),
                "samples": list(s.get('samples', [])),
                "embedding_size": len(s.get("embedding", [])),
            }
    return None


def get_similarity(name1: str, name2: str) -> float:
    """Return cosine similarity between two stored speakers (by name or id)."""
    db = load_db()
    e1 = e2 = None
    t1 = name1.strip().lower()
    t2 = name2.strip().lower()
    for s in db.get("speakers", []):
        sid = (s.get('id') or '').lower()
        sname = s["name"].strip().lower()
        if sid == t1 or sname == t1:
            e1 = s.get("embedding")
        elif sid == t2 or sname == t2:
            e2 = s.get("embedding")
    if e1 and e2:
        return _cosine_similarity(e1, e2)
    return 0.0
