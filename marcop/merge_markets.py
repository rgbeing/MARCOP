import argparse, os, re
from collections import Counter, defaultdict

import numpy as np

from marcop import common as C

THRESHOLD = 0.25
MIN_SIZE = 3
TAIL = {"game", "experience", "title"}
POSITION_GROUPS = [
    {"front", "fore", "forward", "head", "headlight", "bow"},
    {"rear", "back", "hind", "tail", "taillight", "stern"},
    {"left", "port"}, {"right", "starboard"},
    {"upper", "top"}, {"lower", "bottom"},
    {"inner", "inside"}, {"outer", "outside"},
]


def normalize(s):
    s = (s or "").lower()
    s = re.sub(r"single[ \-]?player", "singleplayer", s)
    s = re.sub(r"multi[ \-]?player", "multiplayer", s)
    s = re.sub(r"\bco[ \-]?op\b", "coop", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    t = re.sub(r"\b(\w{3,})s\b", r"\1", " ".join(s.split())).split()
    while len(t) > 1 and t[-1] in TAIL:
        t.pop()
    return " ".join(t)


def position_conflicts(vocab):
    g = {k: {n for n, grp in enumerate(POSITION_GROUPS) if set(v.split()) & grp} for k, v in enumerate(vocab)}
    ks = [k for k in g if g[k]]
    return [(a, b) for x, a in enumerate(ks) for b in ks[x + 1:] if not (g[a] & g[b])]


def cluster(E, threshold, conflicts):
    from sklearn.cluster import AgglomerativeClustering
    if not conflicts:
        return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                       linkage="average").fit(E).labels_
    D = np.clip(1.0 - E.astype(np.float64) @ E.astype(np.float64).T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    a, b = np.array(conflicts).T
    D[a, b] = D[b, a] = 2.0
    return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="precomputed",
                                   linkage="average").fit(D).labels_


def merge(raw, emb_cache=None):
    names = {i: list(dict.fromkeys(c for c in map(normalize, v) if c)) for i, v in raw.items()}
    vocab = sorted({c for v in names.values() for c in v})
    lab = dict(zip(vocab, (int(x) for x in cluster(C.text_embed(vocab, emb_cache), THRESHOLD,
                                                    position_conflicts(vocab)))))
    freq = defaultdict(Counter)
    for v in names.values():
        for c in v:
            freq[lab[c]][c] += 1
    spelling = defaultdict(Counter)
    for v in raw.values():
        for x in v:
            if normalize(x):
                spelling[normalize(x)][x.strip().lower()] += 1
    mem = {i: list(dict.fromkeys(lab[c] for c in v)) for i, v in names.items()}
    size = Counter(m for v in mem.values() for m in v)
    name_of = {m: spelling[f.most_common(1)[0][0]].most_common(1)[0][0] for m, f in freq.items()}
    return {i: [name_of[m] for m in v if size[m] >= MIN_SIZE] for i, v in mem.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    rows = [r for r in C.read_jsonl(os.path.join(a.out, "markets_raw.jsonl")) if not r.get("error")]
    markets = merge({r["id"]: r["out"] for r in rows}, os.path.join(a.out, "cache", "market_names_emb.npz"))
    C.write_jsonl(os.path.join(a.out, "markets.jsonl"),
                  [{"item_id": r["id"], "title": r["title"], "markets": markets[r["id"]]} for r in rows])
    size = Counter(m for v in markets.values() for m in v)
    per = Counter(len(v) for v in markets.values())
    print(f"items {len(markets)}  markets {len(size)}  markets per item {dict(sorted(per.items()))}")


if __name__ == "__main__":
    main()
