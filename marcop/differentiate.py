import argparse, json, os, re
from collections import Counter

from marcop import common as C
from marcop.incorporate import block
from marcop.user_signals import title_of

HEAD = re.compile(r"^\W*(contexts|characteristics)\W*$", re.I)
FIELDS = ("contexts", "characteristics")


def render_item(title, p):
    j = lambda L: "\n".join(f"- {x}" for x in L) or "None."
    return f"Title: {title}\n[Contexts]\n{j(p['contexts'])}\n[Characteristics]\n{j(p['characteristics'])}"


def postprocess(raw, p):
    sec, cur, seen = {f: [] for f in FIELDS}, None, set()
    for line in raw.splitlines():
        s = line.strip()
        m = HEAD.match(s)
        if m:
            cur = m.group(1).lower()
            seen.add(cur)
            continue
        s = s.lstrip("-•* ").strip()
        if cur and s and s.rstrip(".").lower() != "none":
            sec[cur].append(s)
    if not seen:
        return None
    return {f: list(dict.fromkeys(sec[f]))[:len(p[f])] for f in FIELDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--concurrency", type=int, default=256)
    a = ap.parse_args()
    meta = C.load_meta(a.data)
    inc = {r["id"]: r for r in C.read_jsonl(os.path.join(a.out, "incorporated.jsonl"))}
    alt = json.load(open(os.path.join(a.out, "alternatives.json")))
    system, user = C.load_prompt("differentiate")

    jobs = []
    for uid, p in inc.items():
        i, m = p["item"], p["market"]
        comps = [j for j in alt.get(i, {}).get(m, []) if f"{j}|{m}" in inc]
        if not m or not comps or not (p["contexts"] or p["characteristics"]):
            continue
        text = (user.replace("{market}", m).replace("{target}", render_item(title_of(meta[i]), p))
                .replace("{comparisons}", "\n\n".join(f"## Comparison item {n}\n" + render_item(title_of(meta[j]), inc[f"{j}|{m}"])
                                                       for n, j in enumerate(comps, 1)))
                .replace("{n_ctx}", str(len(p["contexts"]))).replace("{n_char}", str(len(p["characteristics"]))))
        jobs.append({"id": uid, "comparisons": comps, "messages": (system, text)})
    raw_path = os.path.join(a.out, "differentiate_raw.jsonl")
    C.run_llm(jobs, raw_path, max_tokens=700, parse=str.strip, concurrency=a.concurrency)

    raw = {r["id"]: r["out"] for r in C.read_jsonl(raw_path) if not r.get("error")}
    sent = {j["id"] for j in jobs}
    rows, cnt = [], Counter()
    for uid, p in inc.items():
        lines = {f: p[f] for f in FIELDS}
        if uid not in sent:
            st = "not_sent"
        elif uid not in raw:
            continue
        else:
            out = postprocess(raw[uid], p)
            lines, st = (out, "llm") if out is not None else (lines, "format_failure")
        rows.append(dict(id=uid, item=p["item"], market=p["market"], **lines,
                         profile=block(p["market"], lines["contexts"], lines["characteristics"])))
        cnt[st] += 1
    C.write_jsonl(os.path.join(a.out, "item_profiles.jsonl"), rows)
    print(f"item profiles {len(rows):,}  " + "  ".join(f"{k} {v:,}" for k, v in cnt.items()))


if __name__ == "__main__":
    main()
