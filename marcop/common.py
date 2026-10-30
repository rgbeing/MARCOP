import asyncio, datetime as dt, html, json, os, sys, time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(ROOT, "prompts")

LLM_MODEL = os.environ.get("MARCOP_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LLM_URL = os.environ.get("MARCOP_LLM_URL", "http://localhost:8000/v1")
TEXT_EMB_MODEL = os.environ.get("MARCOP_TEXT_EMB_MODEL", "BAAI/bge-large-en-v1.5")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def load_meta(data):
    return {str(m["parent_asin"]): m for m in read_jsonl(f"{data}/meta.jsonl")}


def load_users(data):
    return read_jsonl(f"{data}/users.jsonl")


def train_pairs(users):
    return {(u["user_id"], i) for u in users for i in u["history"]}


def review_time(o):
    return dt.datetime.strptime(o["date"], "%Y-%m-%d").timestamp()


def load_reviews(data, pairs):
    best = {}
    with open(f"{data}/reviews.jsonl") as f:
        for n, l in enumerate(f):
            o = json.loads(l)
            k = (o["user_id"], str(o["product_id"]))
            if k not in pairs:
                continue
            t = (review_time(o), n)
            if k not in best or t < best[k]["t"]:
                best[k] = dict(o, t=t)
    return best


def review_text(o):
    return " ".join(html.unescape(o.get("text") or "").split())


def clip_words(s, n):
    if len(s) <= n:
        return s
    cut = s[:n]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip() + " …"


def cf_embeddings(data):
    z = np.load(f"{data}/lightgcn_embeddings.npz", allow_pickle=True)
    ix = {str(k): t for t, k in enumerate(z["item_ids"])}
    uix = {str(k): t for t, k in enumerate(z["user_ids"])}
    return ix, z["item_emb_l2"].astype(np.float64), uix, z["user_emb"].astype(np.float64)


def text_embed(texts, cache=None):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["texts"]) == list(texts):
            return z["E"]
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    E = SentenceTransformer(TEXT_EMB_MODEL, device=device).encode(
        list(texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, E=E, texts=np.array(list(texts), dtype=object))
    return E


def load_prompt(name):
    sysp = open(f"{PROMPTS}/{name}.system.txt").read().strip()
    up = f"{PROMPTS}/{name}.user.txt"
    return sysp, (open(up).read().strip() if os.path.exists(up) else None)


async def _probe_json(client, model, schema):
    msg = [{"role": "user", "content": "Reply in prose, not JSON: name a fruit."}]
    for key, val in (("structured_outputs", {"json": schema}), ("guided_json", schema)):
        try:
            r = await client.chat.completions.create(model=model, messages=msg, max_tokens=100,
                                                     temperature=0.0, extra_body={key: val})
            json.loads(r.choices[0].message.content)
            return {key: val}
        except Exception:
            continue
    sys.exit("The server does not enforce JSON schemas (tried structured_outputs and guided_json).")


def run_llm(jobs, out_path, *, max_tokens, schema=None, parse=None, model=LLM_MODEL, base_url=LLM_URL,
            concurrency=64):
    return asyncio.run(_run_llm(jobs, out_path, max_tokens, schema, parse or (lambda x: x), model, base_url,
                                concurrency))


async def _run_llm(jobs, out_path, max_tokens, schema, parse, model, base_url, concurrency):
    from openai import AsyncOpenAI
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    prev = read_jsonl(out_path) if os.path.exists(out_path) else []
    good = [r for r in prev if not r.get("error")]
    if len(good) < len(prev):
        write_jsonl(out_path, good)
    done = {r["id"] for r in good}
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{os.path.basename(out_path)}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "x"))
    extra = await _probe_json(client, model, schema) if schema else {}
    q = asyncio.Queue()
    for j in todo:
        q.put_nowait(j)
    fout, stat, t0 = open(out_path, "a"), defaultdict(int), time.time()

    async def worker():
        while not q.empty():
            j = q.get_nowait()
            rec = {k: v for k, v in j.items() if k != "messages"}
            sys_msg, user_msg = j["messages"]
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model, temperature=0.0, max_tokens=max_tokens, extra_body=extra,
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}])
                    text = r.choices[0].message.content
                    rec["raw"] = text
                    rec["out"] = parse(json.loads(text) if schema else text)
                    break
                except Exception as e:
                    if attempt == 2:
                        rec["error"] = f"{type(e).__name__}: {e}"[:200]
                        stat["error"] += 1
                    else:
                        await asyncio.sleep(1 + 2 * attempt)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stat["n"] += 1
            if stat["n"] % 1000 == 0:
                fout.flush()
                print(f"  {stat['n']}/{len(todo)}  {stat['n'] / (time.time() - t0):.1f}/s  "
                      f"failed {stat['error']}", flush=True)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    fout.close()
    print(f"  finished in {time.time() - t0:.0f}s, failed {stat['error']}", flush=True)
