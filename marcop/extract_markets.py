import argparse, os
from collections import defaultdict

from marcop import common as C

N_REVIEWS, REVIEW_CHARS = 10, 500
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"markets": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string"}}},
          "required": ["markets"]}


def render_item(m):
    return "\n".join([f'Title: {m.get("title") or m.get("app_name")}',
                      f'Tags (by how many players applied them): {", ".join((m.get("tags") or [])[:14])}',
                      f'Modes: {", ".join(m.get("specs") or [])}'])


def recent_reviews(reviews):
    by_item = defaultdict(list)
    for (_, i), o in reviews.items():
        if C.review_text(o):
            by_item[i].append(o)
    return {i: [C.clip_words(C.review_text(o), REVIEW_CHARS)
                for o in sorted(v, key=lambda o: o["t"], reverse=True)[:N_REVIEWS]]
            for i, v in by_item.items()}


def render(m, revs):
    x = render_item(m)
    if revs:
        x += "\n\nBuyer reviews (most recent first)\n" + "\n".join(f"- {r}" for r in revs)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    meta, users = C.load_meta(a.data), C.load_users(a.data)
    revs = recent_reviews(C.load_reviews(a.data, C.train_pairs(users)))
    system, _ = C.load_prompt("market")
    jobs = [{"id": i, "title": m.get("title") or m.get("app_name") or "",
             "messages": (system, render(m, revs.get(i, [])))} for i, m in meta.items()]
    C.run_llm(jobs, os.path.join(a.out, "markets_raw.jsonl"), max_tokens=100, schema=SCHEMA,
              parse=lambda o: [s for s in o["markets"] if s.strip()], concurrency=a.concurrency)


if __name__ == "__main__":
    main()
