import argparse, os, re
from collections import Counter, defaultdict

from marcop import common as C
from marcop.user_signals import profile_lines

N_BUYERS = 10
CAP = {"contexts": 2, "objective": 5, "subjective": 3}
HEAD = re.compile(r"^\W*(contexts|characteristics)\W*$", re.I)
PREF = re.compile(r"\s*\((?:P\d+(?:\s*,\s*P?\d+)*)\)")
COUNT = re.compile(r"\s*\(\s*\d+\s+(?:of\s+\d+\s+)?buyers?\b[^)]*\)", re.I)
EMPTY = re.compile(r"\b(?:like|dislike)s? (?:none|nothing|no complaints?)\b|\bnone of its\b"
                   r"|\bno (?:complaints?|dislikes?) (?:were |was )?(?:mentioned|noted)", re.I)
SUBJECTIVE = re.compile(r"^buyers\b", re.I)


def block(market, contexts, characteristics):
    ctx = ([f"- Market: {market}"] if market else []) + [f"- {x}" for x in contexts]
    ch = [f"- {x}" for x in characteristics]
    j = lambda L: "\n".join(L) or "None."
    return f"[Contexts]\n{j(ctx)}\n[Characteristics]\n{j(ch)}"


def render_input(plines, buyers):
    cnt = Counter(n for _, s in buyers for x in s["implied"] for n in x["lines"])
    order = sorted(range(1, len(plines) + 1), key=lambda n: (-cnt[n], n))
    new = {old: k for k, old in enumerate(order, 1)}
    prof = "\n".join(f"[P{k}] {plines[old - 1]}" for k, old in enumerate(order, 1))
    blocks = []
    for n, (rating, s) in enumerate(buyers, 1):
        b = [f"[{n}]"] + ([f"  Rating: {rating}"] if rating else [])
        if s["contexts"]:
            b += ["  Contexts:"] + [f"  - {x}" for x in s["contexts"]]
        pref = s["stated"] + [x["text"] + (f" ({', '.join(f'P{new[k]}' for k in x['lines'] if k in new)})"
                                           if any(k in new for k in x["lines"]) else "") for x in s["implied"]]
        if pref:
            b += ["  Preferences:"] + [f"  - {x}" for x in pref]
        blocks.append("\n".join(b))
    info = dict(n_ctx=sum(bool(s["contexts"]) for _, s in buyers), n_st=sum(bool(s["stated"]) for _, s in buyers),
                ordered=[plines[old - 1] for old in order])
    return prof, "\n\n".join(blocks) or "None.", info


def postprocess(raw, info):
    sec, cur = {"contexts": [], "characteristics": []}, None
    for line in raw.splitlines():
        s = line.strip()
        m = HEAD.match(s)
        if m:
            cur = m.group(1).lower()
            continue
        s = s.lstrip("-•* ").strip()
        if not cur or not s or s.rstrip(".").lower() == "none":
            continue
        s = COUNT.sub("", PREF.sub("", s)).strip()
        if s and not EMPTY.search(s):
            sec[cur].append(s)
    obj = [x for x in sec["characteristics"] if not SUBJECTIVE.match(x)][:CAP["objective"]]
    subj = [x for x in sec["characteristics"] if SUBJECTIVE.match(x)][:CAP["subjective"]] if info["n_st"] else []
    return {"contexts": sec["contexts"][:min(CAP["contexts"], info["n_ctx"])], "characteristics": obj + subj}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--concurrency", type=int, default=192)
    a = ap.parse_args()
    sigs = {r["id"]: r for r in C.read_jsonl(os.path.join(a.out, "item_signals.jsonl")) if not r.get("error")}
    usig = C.read_jsonl(os.path.join(a.out, "user_signals.jsonl"))
    reviews = C.load_reviews(a.data, {(r["user"], r["item"]) for r in usig})
    group = defaultdict(list)
    for r in usig:
        group[(r["item"], r["market"] or "")].append(r)
    prompts = {True: C.load_prompt("incorporate"), False: C.load_prompt("incorporate_0")}

    units, jobs = {}, []
    for uid, s in sigs.items():
        i, m = s["item"], s["market"]
        recent = sorted(group.get((i, m), []), key=lambda r: (reviews[(r["user"], i)]["t"][0], r["user"]))[-N_BUYERS:]
        buyers = [(reviews[(r["user"], i)].get("sentiment"), r) for r in recent
                  if r["contexts"] or r["stated"] or r["implied"]]
        prof, buy, info = render_input(profile_lines(s["out"]), buyers)
        units[uid] = dict(item=i, market=m, buyers=len(buyers), **info)
        if buyers:
            system, user = prompts[bool(m)]
            jobs.append({"id": uid, "messages": (system, user.replace("{market}", m).replace("{profile}", prof)
                                                 .replace("{buyers}", buy))})
    raw_path = os.path.join(a.out, "incorporate_raw.jsonl")
    C.run_llm(jobs, raw_path, max_tokens=700, parse=str.strip, concurrency=a.concurrency)

    raw = {r["id"]: r["out"] for r in C.read_jsonl(raw_path) if not r.get("error")}
    rows, cnt = [], Counter()
    for uid, x in units.items():
        if not x["buyers"]:
            lines, st = {"contexts": [], "characteristics": x["ordered"][:CAP["objective"]]}, "no_buyers"
        elif uid in raw:
            lines, st = postprocess(raw[uid], x), "llm"
        else:
            continue
        rows.append(dict(id=uid, item=x["item"], market=x["market"], **lines))
        cnt[st] += 1
    C.write_jsonl(os.path.join(a.out, "incorporated.jsonl"), rows)
    print(f"incorporated profiles {len(rows):,}  " + "  ".join(f"{k} {v:,}" for k, v in cnt.items()))


if __name__ == "__main__":
    main()
