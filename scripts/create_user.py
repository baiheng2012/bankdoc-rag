"""Create an API user and print the key ONCE.

The database stores only the hash, so the key cannot be shown again. If it
is lost, create a new user (or add a rotate feature later).

Usage:
    python scripts/create_user.py --user-id alice --clearance 0
    python scripts/create_user.py --user-id bob --clearance 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg

from bankdoc.api.auth import generate_api_key, hash_api_key
from bankdoc.core.db import close_pool, get_connection


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an API user.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--clearance", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()

    api_key = generate_api_key()
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO api_users (user_id, api_key_hash, clearance) VALUES (%s, %s, %s)",
                (args.user_id, hash_api_key(api_key), args.clearance),
            )
    except psycopg.errors.UniqueViolation:
        print(f"User '{args.user_id}' already exists.")
        sys.exit(1)
    finally:
        close_pool()

    print(f"Created user '{args.user_id}' with clearance {args.clearance}.")
    print(f"API key (shown only once): {api_key}")


if __name__ == "__main__":
    main()
