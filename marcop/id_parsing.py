from __future__ import annotations

import re

_CANARY_LETTERS = frozenset("aqkv")

_ID_BRACKET_RUN_RE = re.compile(r"(?:\[[a-z0-9]{2}\]\s*,\s*){2,}\[[a-z0-9]{2}\]")
_ID_BARE_RUN_RE = re.compile(r"(?:\b[a-z0-9]{2}\b\s*,\s*){2,}\b[a-z0-9]{2}\b")
_ID_TOKEN_RE = re.compile(r"[a-z0-9]{2}")


def parse_id_ranking_detailed(response_text: str, candidate_ids: list[str],
                              label_ids: list[str]) -> tuple[list[str], list[str], int, dict]:
    n = len(candidate_ids)
    id_to_item = dict(zip(label_ids, candidate_ids))
    valid = set(label_ids)
    text = response_text.lower() if response_text else ""

    runs = _ID_BRACKET_RUN_RE.findall(text) or _ID_BARE_RUN_RE.findall(text)
    tokens = _ID_TOKEN_RE.findall(max(runs, key=len)) if runs else []

    raw_ranking_list, seen = [], set()
    n_duplicate = n_hallucinated = n_canary = 0
    for token in tokens:
        if token not in valid:
            n_hallucinated += 1
            if token[0] in _CANARY_LETTERS:
                n_canary += 1
            continue
        if token in seen:
            n_duplicate += 1
            continue
        seen.add(token)
        raw_ranking_list.append(id_to_item[token])

    n_missing_items = n - len(raw_ranking_list)
    picked = set(raw_ranking_list)
    ranking_list = raw_ranking_list + [i for i in candidate_ids if i not in picked]

    diagnostics = {
        "has_run": bool(runs),
        "n_emitted": len(tokens),
        "n_duplicate": n_duplicate,
        "n_hallucinated": n_hallucinated,
        "n_canary": n_canary,
    }
    return ranking_list, raw_ranking_list, n_missing_items, diagnostics
