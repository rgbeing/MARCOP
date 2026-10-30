import argparse, json, os
from collections import Counter, defaultdict

import numpy as np

from marcop import common as C


class Allocator:
    def __init__(self, markets, alternatives, ix, E, uix, U):
        self.uix, self.alt = uix, alternatives
        self.mk = {i: list(dict.fromkeys(ms)) for i, ms in markets.items() if ms}
        rows = defaultdict(list)
        for i, ms in self.mk.items():
            for m in ms:
                rows[m].append(ix[i])
        names = sorted(rows)
        self.col = {m: c for c, m in enumerate(names)}
        M = np.stack([E[rows[m]].mean(0) for m in names])
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        cos = (U / np.linalg.norm(U, axis=1, keepdims=True)) @ M.T
        self.Z = (cos - cos.mean(0)) / cos.std(0)
        self.tau = np.median(self.Z, axis=1)

    def __call__(self, user, item):
        ms = self.mk.get(item)
        if not ms:
            return None, "no_market"
        t = self.uix[user]
        z = [self.Z[t, self.col[m]] for m in ms]
        k = int(np.argmax(z))
        if z[k] < self.tau[t]:
            return None, "below_median"
        if not self.alt.get(item, {}).get(ms[k]):
            return None, "no_alternatives"
        return ms[k], "assigned"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    ix, E, uix, U = C.cf_embeddings(a.data)
    markets = {r["item_id"]: r["markets"] for r in C.read_jsonl(os.path.join(a.out, "markets.jsonl"))}
    phi = Allocator(markets, json.load(open(os.path.join(a.out, "alternatives.json"))), ix, E, uix, U)
    users = C.load_users(a.data)
    for side, key in (("history", "history"), ("candidates", "candidates")):
        rows, cnt = [], Counter()
        for u in users:
            for i in u[key]:
                m, st = phi(u["user_id"], i)
                rows.append({"user": u["user_id"], "item": i, "market": m, "status": st})
                cnt[st] += 1
        C.write_jsonl(os.path.join(a.out, f"allocation_{side}.jsonl"), rows)
        print(f"{side:10s} {len(rows):9,}  " + "  ".join(f"{k} {v / len(rows):.1%}" for k, v in cnt.most_common()))


if __name__ == "__main__":
    main()
