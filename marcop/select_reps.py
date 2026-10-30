import argparse, json, os
from collections import defaultdict

import numpy as np

from marcop import common as C

K = 5


def kmedoids(D, k, seeds=5, iters=50):
    n, best = len(D), (np.inf, None)
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        med = [int(rng.integers(n))]
        while len(med) < k:
            dmin = D[:, med].min(1) ** 2
            med.append(int(rng.choice(n, p=dmin / dmin.sum())) if dmin.sum() > 0 else int(rng.integers(n)))
        for _ in range(iters):
            lab = np.argmin(D[:, med], 1)
            new = []
            for c in range(k):
                mem = np.where(lab == c)[0]
                new.append(med[c] if len(mem) == 0 else int(mem[np.argmin(D[np.ix_(mem, mem)].sum(1))]))
            if new == med:
                break
            med = new
        cost = D[:, med].min(1).sum()
        if cost < best[0] and len(set(med)) == k:
            best = (cost, med)
    return best[1] or list(range(k))


def select(markets, ix, X, k=K):
    members = defaultdict(set)
    for i, ms in markets.items():
        if i in ix:
            for m in ms:
                members[m].add(i)
    reps = {}
    for m, its in members.items():
        its = sorted(its)
        if len(its) <= k:
            reps[m] = its
            continue
        E = X[[ix[i] for i in its]]
        reps[m] = [its[j] for j in kmedoids(1.0 - np.clip(E @ E.T, -1, 1), k)]
    return reps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    ix, X, _, _ = C.cf_embeddings(a.data)
    markets = {r["item_id"]: r["markets"] for r in C.read_jsonl(os.path.join(a.out, "markets.jsonl"))}
    reps = select(markets, ix, X, a.k)
    json.dump(reps, open(os.path.join(a.out, "reps.json"), "w"), ensure_ascii=False, indent=0)
    print(f"markets {len(reps)} -> {os.path.join(a.out, 'reps.json')}")


if __name__ == "__main__":
    main()
