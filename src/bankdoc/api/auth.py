"""API-key authentication.

Flow: the client sends its key in the X-API-Key header. We hash it, look
the hash up in api_users, and return the user's identity and clearance.
get_current_user() is a FastAPI "dependency": any endpoint that declares
Depends(get_current_user) is automatically protected.
"""

import hashlib
import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from bankdoc.core.db import get_connection


@dataclass
class User:
    """The authenticated caller, as seen by the rest of the application."""

    user_id: str
    clearance: int  # 0 public, 1 internal, 2 restricted


def generate_api_key() -> str:
    """Create a new random API key.

    secrets.token_urlsafe uses the operating system's cryptographically
    secure random generator. 32 bytes gives 256 bits of entropy, so the key
    cannot be guessed. The "bk_" prefix only makes keys easy to recognise.
    """
    return "bk_" + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Return the SHA-256 hex digest of an API key.

    Plain SHA-256 is acceptable here because our keys are long random
    values. (Human-chosen PASSWORDS need a slow salted hash such as bcrypt,
    because they are guessable.)
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def find_user_by_key(api_key: str) -> User | None:
    """Look up an active user by API key. Returns None if the key is unknown or revoked."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT user_id, clearance FROM api_users "
            "WHERE api_key_hash = %s AND is_active",
            (hash_api_key(api_key),),
        ).fetchone()
    return User(user_id=row[0], clearance=row[1]) if row else None


def get_current_user(x_api_key: str | None = Header(default=None)) -> User:
    """FastAPI dependency: authenticate the request or raise 401.

    FastAPI maps the parameter name x_api_key to the HTTP header X-API-Key
    (underscores become hyphens). Header(default=None) makes a missing
    header reach our code, so we can return a clean 401 instead of 422.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )
    if not x_api_key:
        raise unauthorized
    user = find_user_by_key(x_api_key)
    if user is None:
        raise unauthorized
    return user
