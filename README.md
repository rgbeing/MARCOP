# MARCOP

Code for **It's All Relative: Market-Based Comparative Profiling for LLM-Based Recommenders**.

This implementation is for Steam dataset.

# How to Run

## Requirements

- Python >= 3.10
- An OpenAI-compatible LLM server. We primarily use [vLLM](https://github.com/vllm-project/vllm) with
  `meta-llama/Llama-3.1-8B-Instruct`.
- A GPU is recommended for the text embedder (`BAAI/bge-large-en-v1.5`).

```bash
pip install -r requirements.txt
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

The server and models are set by environment variables:

| Variable | Default |
|---|---|
| `MARCOP_LLM_URL` | `http://localhost:8000/v1` |
| `MARCOP_LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` |
| `MARCOP_TEXT_EMB_MODEL` | `BAAI/bge-large-en-v1.5` |

## Input data

The scripts read one dataset folder (`--data`) with the following files. Item ids are strings.

| File | One line / entry per | Fields used |
|---|---|---|
| `users.jsonl` | user | `user_id`; `history` (train items, oldest first); `valid`; `test`; `candidates` (20 test candidates including `test`, in the order shown to the reranker) |
| `meta.jsonl` | item | `parent_asin` (item id), `title` or `app_name`, `genres`, `tags`, `specs`, `developer`, `publisher`, `price` |
| `reviews.jsonl` | review | `user_id`, `product_id` (item id), `text`, `date` (`YYYY-MM-DD`), `sentiment`, `hours` |
| `lightgcn_embeddings.npz` | — | `user_ids`, `user_emb`, `item_ids`, `item_emb_l2` (L2-normalized item embeddings) from a LightGCN model trained on the train interactions only |

One line of each JSON Lines file (other fields are ignored). The lines come from the Steam data, except that the
user id is replaced with a placeholder and the reviewer's username is removed:

`users.jsonl`
```json
{"user_id": "76561198000000001", "history": ["250620", "558420", "554620"], "valid": "207610", "test": "35140", "candidates": ["282140", "271590", "418370", "377160", "379720", "614570", "319630", "35140", "414340", "501300", "612880", "601430", "524220", "374320", "268910", "424840", "305620", "383870", "343860", "462770"]}
```

`meta.jsonl`
```json
{"publisher":"Devolver Digital","genres":["Adventure","Indie"],"app_name":"Stories Untold","sentiment":"Very Positive","title":"Stories Untold","release_date":"2017-02-27","tags":["Adventure","Indie","Text-Based","Horror","Atmospheric","Psychological Horror","Singleplayer","Point & Click","Thriller","Sci-fi","Retro"],"specs":["Single-player","Steam Achievements","Steam Cloud"],"price":9.99,"metascore":81,"early_access":false,"parent_asin":"558420","developer":"No Code"}
```

`reviews.jsonl`
```json
{"text":"An interesting narrative experience clearly inspired by stranger things.\nQuite short(2-3 hours) but very polished.\nA mix of simple text games, puzzles and walking simulator sequences.\nOverall an enjoyable experience.","hours":2.9,"date":"2017-02-28","early_access":false,"user_id":"76561198000000001","product_id":"558420","page_order":6,"products":577,"page":128,"sentiment":"Positive"}
```


## Running

Run the steps in order. Each one reads the previous outputs from `--out` and writes its own. LLM steps can be
resumed: rerunning a step skips the calls that already finished.

```bash
DATA=path/to/Steam
OUT=outputs

python -m marcop.extract_markets --data $DATA --out $OUT   # Eq. (3)        LLM
python -m marcop.merge_markets               --out $OUT   # §4.1.1 normalization
python -m marcop.select_reps     --data $DATA --out $OUT   # Eq. (4)
python -m marcop.item_signals    --data $DATA --out $OUT   # Eqs. (5)–(7)   LLM
python -m marcop.allocate        --data $DATA --out $OUT   # Eqs. (8)–(9)
python -m marcop.user_signals    --data $DATA --out $OUT   # Eq. (10)       LLM
python -m marcop.user_profiles   --data $DATA --out $OUT   # Eq. (13)       LLM
python -m marcop.incorporate     --data $DATA --out $OUT   # Eq. (11)       LLM
python -m marcop.differentiate   --data $DATA --out $OUT   # Eq. (12)       LLM
python -m marcop.rerank          --data $DATA --out $OUT   # Eq. (2)        LLM
```

`rerank` prints HR@5, nDCG@5, HR@10 and nDCG@10 of the test item over all users. To rerank more than once, pass
`--run 2`, `--run 3`, …; each run is stored separately. LLM steps take `--concurrency` (number of parallel requests).

## Outputs

| Step | File | Content |
|---|---|---|
| `extract_markets` | `markets_raw.jsonl` | raw market names M̃_i per item |
| `merge_markets` | `markets.jsonl` | markets M_i per item |
| `select_reps` | `reps.json` | market representatives C_m |
| `item_signals` | `alternatives.json` | proxy alternative sets C_{i,m} |
| | `item_signals.jsonl` | item-market signals S_i(m) and market-agnostic signals S_i(0) |
| `allocate` | `allocation_history.jsonl`, `allocation_candidates.jsonl` | φ(u, i) for train and candidate pairs |
| `user_signals` | `user_signals.jsonl` | user signals S_{u,i} |
| `user_profiles` | `user_profiles.jsonl` | user profiles P_u |
| `incorporate` | `incorporated.jsonl` | item-market profiles after incorporation P̃_i(m), and P_i(0) |
| `differentiate` | `item_profiles.jsonl` | final item profiles P_i(m) and P_i(0), as shown to the reranker |
| `rerank` | `rerank_r<run>.jsonl` | rank of the test item and HR / nDCG per user |

Files ending in `_raw.jsonl` hold the raw LLM responses of each step.

# Prompts

All prompts are in [`prompts/`](prompts). Table 7 of the paper gives their core instructions.

| Prompt | File |
|---|---|
| p_market | `market.system.txt` |
| p_item-sig, p_item-0 | `item_signal.*`, `item_signal_0.*` |
| p_user-sig | `user_signal.*` |
| p_incorporate | `incorporate.*` (`incorporate_0.*` for P_i(0)) |
| p_diff | `differentiate.*` |
| p_user | `user_profile.system.txt` |
| p_rerank | `rerank.*` |
