"""Hashing for safe-words and challenge answers (PBKDF2-SHA256, stdlib only)."""

import hashlib
import hmac
import re
import secrets
import unicodedata

ITERATIONS = 120_000


def normalize_secret(value: str) -> str:
    """Lowercase, trim, drop punctuation and collapse spaces, so "Pizza Place!" == "pizza place".
    This forgives how a stressed person types what they heard on a call."""
    text = unicodedata.normalize("NFKC", value).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def hash_secret(value: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", normalize_secret(value).encode(), salt.encode(), ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt}${digest.hex()}"


def verify_secret(value: str, stored: str) -> bool:
    try:
        _, iterations, salt, expected = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", normalize_secret(value).encode(), salt.encode(), int(iterations)
    )
    return hmac.compare_digest(digest.hex(), expected)
