from __future__ import annotations

import random
import string

LETTERS = string.ascii_lowercase
DIGITS = string.digits

BANNED_LETTERS = frozenset("aqkv")

ITEM_LETTERS = tuple(c for c in LETTERS if c not in BANNED_LETTERS)
ID_POOL = tuple(c + d for c in ITEM_LETTERS for d in DIGITS)

MAX_CANDIDATES = len(ITEM_LETTERS)


def assign_ids(user_id: str, n: int) -> list[str]:
    if n > MAX_CANDIDATES:
        raise ValueError(
            f"{n} candidates exceeds {MAX_CANDIDATES} available first letters "
            f"(26 minus the reserved {''.join(sorted(BANNED_LETTERS))})")
    rng = random.Random(f"{user_id}|v17")
    letters = rng.sample(ITEM_LETTERS, n)
    return [rng.choice([c + d for d in DIGITS]) for c in letters]


def is_canary(item_id: str) -> bool:
    return bool(item_id) and item_id[0] in BANNED_LETTERS
