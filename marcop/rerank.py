import argparse, html, math, os

from marcop import common as C
from marcop.id_parsing import parse_id_ranking_detailed
from marcop.item_ids import assign_ids

KS = (5, 10)


def title_of(m):
    return " ".join(html.unescape(m.get("title") or m.get("app_name") or "").split())


def build_prompt(user, uprof, profile_of, titles, template):
    cands = list(user["candidates"])
    ids = assign_ids(user["user_id"], len(cands))
    blocks = "\n\n".join(f"[{lid}] {titles[i]}\n{profile_of(i)}" for lid, i in zip(ids, cands))
    return template.replace("{user_profile}", uprof).replace("{candidate_items}", blocks), cands, ids


def metrics(rank):
    return {**{f"hr@{k}": int(rank <= k) for k in KS},
            **{f"ndcg@{k}": (1 / math.log2(rank + 1) if rank <= k else 0.0) for k in KS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--run", type=int, default=1, help="index of a repeated reranking (results are kept apart)")
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args()
    meta, users = C.load_meta(a.data), C.load_users(a.data)
    titles = {i: title_of(m) for i, m in meta.items()}
    uprof = {r["user"]: r["profile"] for r in C.read_jsonl(os.path.join(a.out, "user_profiles.jsonl"))}
    iprof = {r["id"]: r["profile"] for r in C.read_jsonl(os.path.join(a.out, "item_profiles.jsonl"))}
    phi = {(r["user"], r["item"]): r["market"]
           for r in C.read_jsonl(os.path.join(a.out, "allocation_candidates.jsonl"))}
    system, template = C.load_prompt("rerank")

    built, jobs = {}, []
    for u in users:
        uid = u["user_id"]
        if uid not in uprof:
            continue
        profile_of = lambda i, uid=uid: iprof[f"{i}|{phi.get((uid, i)) or ''}"]
        text, cands, ids = build_prompt(u, uprof[uid], profile_of, titles, template)
        built[uid] = (cands, ids, u["test"])
        jobs.append({"id": uid, "messages": (system, text)})
    raw_path = os.path.join(a.out, f"rerank_r{a.run}_raw.jsonl")
    C.run_llm(jobs, raw_path, max_tokens=700, parse=lambda x: x, concurrency=a.concurrency)

    rows = []
    for r in C.read_jsonl(raw_path):
        if r.get("error"):
            continue
        cands, ids, test = built[r["id"]]
        ranking, _, n_missing, diag = parse_id_ranking_detailed(r["out"], cands, ids)
        rank = ranking.index(test) + 1
        rows.append({"user": r["id"], "rank": rank, **metrics(rank), "n_missing_items": n_missing,
                     "has_run": diag["has_run"]})
    C.write_jsonl(os.path.join(a.out, f"rerank_r{a.run}.jsonl"), rows)
    n = len(rows)
    print(f"users {n:,} (of {len(users):,})  " +
          "  ".join(f"{m}@{k} {sum(x[f'{m}@{k}'] for x in rows) / max(n, 1):.4f}" for k in KS for m in ("ndcg", "hr")))


if __name__ == "__main__":
    main()
