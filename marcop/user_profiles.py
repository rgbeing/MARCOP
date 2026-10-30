import argparse, os, re
from collections import Counter

from marcop import common as C
from marcop.user_signals import title_of

MAX_ITEMS, MAX_MARKETS = 50, 5
CAP = {"contexts": 3, "stated": 4, "implied": 3}
HEAD = re.compile(r"^\W*(contexts|preferences)\W*$", re.I)
IMPLIED = re.compile(r"^it seems (that )?they\b", re.I)
NUMS = re.compile(r"\s*\(\s*#?\d+(?:\s*[,;]\s*#?\d+)*\s*\)\s*\.?\s*$")
CITE = re.compile(r"\s*\(\s*#\d+(?:\s*(?:[,;]|and)\s*#?\d+)*\s*\)|\s*#\d+")


def render_user(items, alloc, sigs, ratings, meta):
    items = items[-MAX_ITEMS:]
    blocks, info, mk = [], dict(n_evid=0, n_ctx=0, impl=[]), {}
    for n, i in enumerate(items, 1):
        m, s = alloc.get(i), sigs.get(i) or {}
        lines = [f"[{n}] {title_of(meta[i])}"]
        if m:
            lines.append(f"  Market: {m}")
            c, _ = mk.get(m, (0, 0))
            mk[m] = (c + 1, n)
        if ratings.get(i):
            lines.append(f"  Rating: {ratings[i]}")
        ctx, pref = s.get("contexts") or [], (s.get("stated") or []) + [x["text"] for x in s.get("implied") or []]
        if ctx:
            lines += ["  Contexts:"] + [f"  - {x}" for x in ctx]
            info["n_ctx"] += 1
        if pref:
            lines += ["  Preferences:"] + [f"  - {x}" for x in pref]
        if s.get("implied"):
            info["impl"].append(n)
        if ctx or pref:
            info["n_evid"] += 1
        blocks.append("\n".join(lines))
    info["markets"] = sorted(mk, key=lambda k: (-mk[k][0], -mk[k][1]))[:MAX_MARKETS]
    info["bought"] = [f"  {n}. {title_of(meta[i])}" + (f" ({alloc[i]})" if alloc.get(i) else "")
                      for n, i in enumerate(items, 1)]
    return "\n\n".join(blocks), info


def strip_cites(s):
    groups = CITE.findall(s) + ([NUMS.search(s).group(0)] if NUMS.search(s) else [])
    nums = sorted({int(x) for g in groups for x in re.findall(r"\d+", g)})
    body = NUMS.sub("", CITE.sub("", s)) if groups else s
    body = re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s{2,}", " ", body)).strip(" ,;")
    if body and body[-1] not in ".!?":
        body += "."
    return body, nums


def postprocess(raw, info):
    out, cur = {"contexts": [], "stated": [], "implied": []}, None
    for line in raw.splitlines():
        s = line.strip()
        m = HEAD.match(s)
        if m:
            cur = m.group(1).lower()
            continue
        s = s.lstrip("-•* ").strip()
        if not cur or not s or s.rstrip(".").lower() == "none":
            continue
        body, nums = strip_cites(s)
        if cur == "contexts":
            out["contexts"].append(body)
        elif IMPLIED.match(body):
            good = [x for x in nums if x in set(info["impl"])]
            if good:
                out["implied"].append((body, good))
        else:
            out["stated"].append(body)
    out["contexts"] = out["contexts"][:min(CAP["contexts"], info["n_ctx"])]
    out["stated"] = out["stated"][:CAP["stated"]]
    out["implied"] = [b for b, _ in sorted(out["implied"], key=lambda x: (-len(x[1]), -max(x[1])))[:CAP["implied"]]]
    return out


def render_profile(lines, info):
    mk = [f"- Markets: {', '.join(info['markets'])}"] if info["markets"] else []
    j = lambda L: "\n".join(L) or "None."
    if lines is None:
        return f"[Contexts]\n{j(mk + ['- Bought (in this order):'] + info['bought'])}\n[Preferences]\nNone."
    ctx = mk + [f"- {x}" for x in lines["contexts"]]
    pref = [f"- {x}" for x in lines["stated"] + lines["implied"]]
    return f"[Contexts]\n{j(ctx)}\n[Preferences]\n{j(pref)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--concurrency", type=int, default=192)
    a = ap.parse_args()
    meta, users = C.load_meta(a.data), C.load_users(a.data)
    alloc, sigs = {}, {}
    for r in C.read_jsonl(os.path.join(a.out, "allocation_history.jsonl")):
        alloc.setdefault(r["user"], {})[r["item"]] = r["market"]
    for r in C.read_jsonl(os.path.join(a.out, "user_signals.jsonl")):
        sigs.setdefault(r["user"], {})[r["item"]] = r
    reviews = C.load_reviews(a.data, C.train_pairs(users))
    system, _ = C.load_prompt("user_profile")

    inputs = {}
    for u in users:
        uid = u["user_id"]
        ratings = {i: reviews[(uid, i)].get("sentiment") for i in u["history"] if (uid, i) in reviews}
        inputs[uid] = render_user(u["history"], alloc.get(uid, {}), sigs.get(uid, {}), ratings, meta)
    jobs = [{"id": uid, "messages": (system, text)} for uid, (text, info) in inputs.items() if info["n_evid"]]
    raw_path = os.path.join(a.out, "user_profiles_raw.jsonl")
    C.run_llm(jobs, raw_path, max_tokens=1000, parse=str.strip, concurrency=a.concurrency)

    raw = {r["id"]: r["out"] for r in C.read_jsonl(raw_path) if not r.get("error")}
    rows, cnt = [], Counter()
    for uid, (_, info) in inputs.items():
        if not info["n_evid"]:
            lines, st = None, "no_signal"
        elif uid in raw:
            lines, st = postprocess(raw[uid], info), "llm"
        else:
            continue
        rows.append({"user": uid, "profile": render_profile(lines, info)})
        cnt[st] += 1
    C.write_jsonl(os.path.join(a.out, "user_profiles.jsonl"), rows)
    print(f"user profiles {len(rows):,}  " + "  ".join(f"{k} {v:,}" for k, v in cnt.items()))


if __name__ == "__main__":
    main()
