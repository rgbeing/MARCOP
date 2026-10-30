import argparse, json, os

from marcop import common as C

TAU_DIST = 0.97


def render_item(o):
    L = [f"Title: {o.get('title') or o.get('app_name')}"]
    dev, pub = o.get("developer"), o.get("publisher")
    if dev:
        L.append(f"Developer: {dev}" + (f"  (Publisher: {pub})" if pub and pub != dev else ""))
    for k, lab in [("genres", "Genres"), ("tags", "Tags (most-agreed first)"), ("specs", "Modes/Features")]:
        v = o.get(k) or []
        if v:
            L.append(f"{lab}: {', '.join(v[:20])}")
    if o.get("price") is not None:
        L.append(f"Price: {o['price']}")
    return "\n".join(L)


def text_of(o):
    parts = [o.get("title") or o.get("app_name") or "", ", ".join(o.get("genres") or []),
             ", ".join(o.get("tags") or []), o.get("publisher") or ""]
    return " || ".join(p for p in parts if p)


def alternatives(markets, reps, ids, T, tau=TAU_DIST):
    row = {i: k for k, i in enumerate(ids)}
    alt = {}
    for i, ms in markets.items():
        for m in ms:
            cs = [j for j in reps.get(m, []) if j != i and float(T[row[i]] @ T[row[j]]) <= tau]
            if cs:
                alt.setdefault(i, {})[m] = cs
    return alt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    meta = C.load_meta(a.data)
    markets = {r["item_id"]: r["markets"] for r in C.read_jsonl(os.path.join(a.out, "markets.jsonl"))}
    reps = json.load(open(os.path.join(a.out, "reps.json")))
    ids = sorted(meta)
    T = C.text_embed([text_of(meta[i]) for i in ids], os.path.join(a.out, "cache", "item_text_emb.npz"))
    alt = alternatives(markets, reps, ids, T)
    json.dump(alt, open(os.path.join(a.out, "alternatives.json"), "w"), ensure_ascii=False)

    sig_sys, sig_usr = C.load_prompt("item_signal")
    zero_sys, zero_usr = C.load_prompt("item_signal_0")
    jobs = []
    for i, ms in alt.items():
        for m, cs in ms.items():
            jobs.append({"id": f"{i}|{m}", "item": i, "market": m, "contrast": cs,
                         "messages": (sig_sys, sig_usr.replace("{m}", m).replace("{t}", render_item(meta[i]))
                                      .replace("{c}", "\n\n".join(render_item(meta[j]) for j in cs)))})
    for i in ids:
        jobs.append({"id": f"{i}|", "item": i, "market": "", "contrast": [],
                     "messages": (zero_sys, zero_usr.replace("{t}", render_item(meta[i])))})
    C.run_llm(jobs, os.path.join(a.out, "item_signals.jsonl"), max_tokens=400, parse=str.strip,
              concurrency=a.concurrency)


if __name__ == "__main__":
    main()
