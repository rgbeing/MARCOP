import argparse, html, os, re
from collections import Counter

from marcop import common as C

REVIEW_CHARS = 2000
CAP = {"contexts": 2, "stated": 3, "implied": 2}


def render_review(o):
    txt = html.unescape(o.get("text") or "").strip()
    txt = txt[:REVIEW_CHARS] + (" …" if len(txt) > REVIEW_CHARS else "")
    h = o.get("hours")
    head = f"Rating: {o.get('sentiment')}" + (f" · {h:g} hours played" if isinstance(h, (int, float)) else "")
    return f"{head}\n{txt}"


def profile_lines(signal):
    return [x.strip() for x in html.unescape(signal).split("\n") if x.strip()]


def render_profile(market, lines):
    body = "\n".join(f"[P{n}] {x}" for n, x in enumerate(lines, 1))
    return f"Market: {market}\n{body}" if market else body


def title_of(m):
    return html.unescape(m.get("title") or m.get("app_name") or "")


TOK = re.compile(r"[a-z0-9']+")
QUOTE = re.compile(r"[\"“]([^\"”]{2,}?)[\"”]")
TRAIL = re.compile(r"\s*\(?\s*(?:[\"“][^\"”]*[\"”]\s*[,;]?\s*)+\)?\s*[.]?\s*$")
INLINE_Q = re.compile(r"\s*\((?:[^()]*?[\"“][^\"”]*[\"”][^()]*?)\)")
PNUM = re.compile(r"\s*\(\s*P?\s*\d+(?:\s*(?:[,;]|and)\s*P?\s*\d+)*\s*\)|\bP\d+\b", re.I)
REASON = re.compile(r",?\s+(as (they|the customer|it)|which (implies|suggests|means)|since|because)\b.*$", re.I)
HEAD = re.compile(r"^\W*(contexts|preferences)\W*$", re.I)
IMPLIED = re.compile(r"^it seems (that )?they\b", re.I)
STOP = set("a an the and or but so to of in on at for with from by as is are was were be been it its it's this that these "
           "those i i'm me my we our you your he she they them their his her very really so just too also all any some much "
           "more most than then there here what which who when where how not no yes do does did have has had will would can "
           "could get got one ones thing things".split())
GENERIC = set("great good nice awesome excellent amazing perfect best better fine ok okay decent cool wonderful fantastic "
              "superb love loved loves like liked likes enjoy enjoyed fun happy pleased satisfied recommend recommended "
              "recommendation product products item items game games buy bought purchase purchased star stars five 5 "
              "thanks thank works worked work working well job quality described expected advertised".split())


def norm(s):
    return " ".join(TOK.findall(s.lower().replace("’", "'")))


def content_words(s):
    return [w for w in TOK.findall(s.lower().replace("’", "'")) if w not in STOP]


def stems(s):
    return {w[:5] for w in content_words(s) if w not in GENERIC and len(w) > 2}


def found(q, text_n, text_t):
    qn = norm(q)
    if not qn:
        return False
    if qn in text_n:
        return True
    qt = qn.split()
    return len(qt) >= 2 and sum(t in text_t for t in qt) / len(qt) >= 0.8


def quoted_line(s, review, body, title):
    qs = QUOTE.findall(s)
    ok = [q for q in qs if found(q, *review) and found(q, *body)]
    tw = set(content_words(title))
    ok = [q for q in ok if any(w not in GENERIC and w not in tw for w in content_words(q))]
    if not ok:
        return None
    x = TRAIL.sub("", INLINE_Q.sub("", s))
    x = re.sub(r"\s*\([^)]*$", "", x).replace("“", "").replace("”", "").replace('"', "")
    return re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s{2,}", " ", x)).strip(" ,;:-—")


def implied_line(s, plines):
    nums = sorted({int(x) for g in PNUM.findall(s) for x in re.findall(r"\d+", g)})
    if not nums or any(not 1 <= x <= len(plines) for x in nums):
        return None
    x = re.sub(r"\s{2,}", " ", PNUM.sub("", s)).strip(" ,;:-—.")
    x = re.sub(r"\s*\([^()]*\)\s*$", "", x).strip(" ,;:-—.")
    x = REASON.sub("", x).strip(" ,;:-—.").replace("“", "").replace("”", "").replace('"', "")
    cited = set().union(*[stems(plines[k - 1]) for k in nums])
    if not stems(re.sub(r"^it seems (that )?they\s+\w+", "", x, flags=re.I)) & cited:
        return None
    if not IMPLIED.match(x):
        x = "It seems that they " + re.sub(r"^(the customer|this customer|they)\s+", "", x, flags=re.I)
    return x, nums


def verify(raw, review, plines, title):
    rn = norm(review)
    head, _, rest = review.partition("\n")
    bn = norm(rest) if head.startswith("Rating:") else rn
    rv, bd = (rn, set(rn.split())), (bn, set(bn.split()))
    out, st, seen, cur = {"contexts": [], "stated": [], "implied": []}, Counter(), set(), None
    for line in raw.splitlines():
        s = line.strip()
        m = HEAD.match(s)
        if m:
            cur = m.group(1).lower()
            continue
        s = s.lstrip("-•* ").strip()
        if not cur or not s or norm(s) == "none":
            continue
        st["lines"] += 1
        if cur == "contexts":
            kind, r = "contexts", quoted_line(s, rv, bd, title)
        elif IMPLIED.match(s) or (PNUM.search(s) and not QUOTE.search(s)):
            kind, r = "implied", implied_line(s, plines)
        else:
            kind, r = "stated", quoted_line(s, rv, bd, title)
        if not r:
            st[f"{kind}_unbacked"] += 1
            continue
        text, nums = r if kind == "implied" else (r, None)
        if text[-1] not in ".!?":
            text += "."
        key = norm(re.sub(r"^it seems (that )?", "", text, flags=re.I))
        if key in seen or len(out[kind]) >= CAP[kind]:
            st["dropped"] += 1
            continue
        seen.add(key)
        out[kind].append({"text": text, "lines": nums} if kind == "implied" else text)
        st["kept"] += 1
    return out, st


def render_signal(sig):
    pref = sig["stated"] + [x["text"] for x in sig["implied"]]
    block = lambda xs: "\n".join(f"- {x}" for x in xs) if xs else "None."
    return f"[Contexts]\n{block(sig['contexts'])}\n[Preferences]\n{block(pref)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--concurrency", type=int, default=192)
    a = ap.parse_args()
    meta = C.load_meta(a.data)
    alloc = {(r["user"], r["item"]): r["market"] for r in C.read_jsonl(os.path.join(a.out, "allocation_history.jsonl"))}
    sigs = {r["id"]: r["out"] for r in C.read_jsonl(os.path.join(a.out, "item_signals.jsonl")) if not r.get("error")}
    reviews = C.load_reviews(a.data, set(alloc))
    system, user = C.load_prompt("user_signal")

    inputs = {}
    for (u, i), m in alloc.items():
        if (u, i) not in reviews:
            continue
        key = f"{i}|{m}" if m else f"{i}|"
        plines = profile_lines(sigs.get(key, ""))
        inputs[f"{u}|{i}"] = dict(user=u, item=i, market=m, plines=plines, review=render_review(reviews[(u, i)]),
                                  title=title_of(meta[i]))
    jobs = [{"id": k, "messages": (system, user.replace("{title}", x["title"])
                                   .replace("{profile}", render_profile(x["market"], x["plines"]))
                                   .replace("{review}", x["review"]))}
            for k, x in inputs.items()]
    raw_path = os.path.join(a.out, "user_signals_raw.jsonl")
    C.run_llm(jobs, raw_path, max_tokens=300, parse=str.strip, concurrency=a.concurrency)

    rows, stat = [], Counter()
    for r in C.read_jsonl(raw_path):
        if r.get("error"):
            continue
        x = inputs[r["id"]]
        sig, st = verify(r["out"], x["review"], x["plines"], x["title"])
        stat.update(st)
        rows.append(dict(user=x["user"], item=x["item"], market=x["market"], **sig, signal=render_signal(sig)))
    C.write_jsonl(os.path.join(a.out, "user_signals.jsonl"), rows)
    print(f"user signals {len(rows):,}  lines {stat['lines']:,}  kept {stat['kept']:,}  " +
          "  ".join(f"{k} {v:,}" for k, v in sorted(stat.items()) if k not in ("lines", "kept")))


if __name__ == "__main__":
    main()
