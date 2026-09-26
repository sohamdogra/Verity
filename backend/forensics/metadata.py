"""Container & file-system forensics: RIFF chunk structure, encoder tags, header consistency,
known synthesis-tool signatures and MAC (modified/accessed/created) timestamps.

Report-only by design: in the HEARSAY test set every clip was re-written by the same FFmpeg
build (encoder tag Lavf58.29.100), while the training files come from other tools. Feeding
these values to the classifier would teach it "which tool wrote the file", not "is it fake".
"""

import os
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

GENERATOR_SIGNATURES = re.compile(
    r"elevenlabs|resemble|play\.?ht|murf|speechify|descript|coqui|xtts|tortoise|bark|vall-?e|"
    r"openvoice|rvc|so-vits|tts|voice ?clone|synthes", re.I)


def _riff_chunks(data: bytes) -> tuple[list[tuple[str, int]], dict[str, str], str | None]:
    """Chunk ids/sizes, LIST/INFO tags, and a structural problem description if any."""
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RIFX") or data[8:12] != b"WAVE":
        return [], {}, None
    chunks, tags, problem = [], {}, None
    declared = struct.unpack("<I", data[4:8])[0] + 8
    if declared != len(data):
        problem = f"RIFF header declares {declared} bytes but the file has {len(data)}"
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4].decode("latin-1")
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        chunks.append((cid, size))
        body = data[pos + 8 : pos + 8 + size]
        if cid == "LIST" and body[:4] == b"INFO":
            sub = 4
            while sub + 8 <= len(body):
                key = body[sub : sub + 4].decode("latin-1")
                n = struct.unpack("<I", body[sub + 4 : sub + 8])[0]
                tags[key] = body[sub + 8 : sub + 8 + n].rstrip(b"\x00").decode("utf-8", "replace")
                sub += 8 + n + (n & 1)
        if cid == "data" and pos + 8 + size > len(data):
            problem = problem or f"data chunk claims {size} bytes but only {len(data) - pos - 8} are present"
        pos += 8 + size + (size & 1)
    return chunks, tags, problem


def _mac_times(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    st = os.stat(path)
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    created = getattr(st, "st_birthtime", st.st_ctime)
    return {"modified": iso(st.st_mtime), "accessed": iso(st.st_atime), "created": iso(created),
            "modified_before_created": st.st_mtime + 1 < created,
            "in_future": max(st.st_mtime, created) > datetime.now(timezone.utc).timestamp() + 60}


def inspect(data: bytes, filename: str, container: str | None, codec: str | None,
            path: Path | None = None) -> tuple[dict, dict]:
    chunks, tags, problem = _riff_chunks(data)
    mac = _mac_times(path)
    signature = GENERATOR_SIGNATURES.search(" ".join(tags.values()))
    notes = []
    score = None
    if signature:
        notes.append(f'Metadata names a speech-synthesis tool ("{signature.group(0)}").')
        score = 0.9
    if problem:
        notes.append(f"Header inconsistency: {problem}, a sign the file was edited or rebuilt.")
        score = max(score or 0, 0.4)
    software = tags.get("ISFT")
    if software:
        notes.append(f"Written by {software}.")
    if mac and (mac["modified_before_created"] or mac["in_future"]):
        notes.append("File timestamps are inconsistent (modified before created, or in the future).")
    if not notes:
        notes.append("Container structure is consistent and carries no tool signatures. "
                     "Metadata is easy to strip or spoof, so a clean result proves nothing.")
    details = {"riff_chunks": [c for c, _ in chunks], "tags": tags, "mac_times": mac,
               "container": container, "codec": codec, "problem": problem}
    technique = {"id": "meta:container", "kind": "metadata", "name": "Container & metadata",
                 "score": score, "finding": " ".join(notes)}
    return technique, details
