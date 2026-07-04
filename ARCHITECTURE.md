# How chat_with_optimusKG answers a question

This document explains how a natural-language question gets turned into an answer grounded
in the OptimusKG graph: what format the graph is stored in, how the LLM is wired to it, and
the exact reasoning loop that runs on every message.

## 1. The KG format

OptimusKG ships as two flat Apache Parquet files:

**`nodes.parquet`** — one row per entity:

| column | type | contents |
|---|---|---|
| `id` | string | unique id, e.g. `ENSG00000000971`, `MONDO_0005180`, `CHEMBL1000` |
| `label` | string | entity type code: `GEN`, `DIS`, `DRG`, `PHE`, `ANA`, `BPO`, `MFN`, `CCO`, `PWY`, `EXP` |
| `properties` | string (JSON) | type-specific attributes — genes carry `symbol`/`biotype`/genomic coordinates, diseases/drugs carry `name`/`description`/synonyms, etc. |

**`edges.parquet`** — one row per relationship:

| column | type | contents |
|---|---|---|
| `from`, `to` | string | node ids the edge connects |
| `label` | string | entity-pair type, e.g. `DIS-GEN`, `DRG-DRG` |
| `relation` | string | the actual relation, e.g. `ASSOCIATED_WITH`, `INDICATION`, `INTERACTS_WITH` |
| `undirected` | bool | whether direction is meaningful |
| `properties` | string (JSON) | evidence scores, source databases, counts, etc. |

Both files together are ~326 MB and encode the full graph: 190K nodes, 21.8M edges.
There's no vector index and no graph database in this setup — it's plain columnar data
queried directly.

Everything below the flat files is DuckDB running SQL over them in-process (`kg.py`).
DuckDB reads Parquet natively and can do substring search, joins, and JSON extraction
(`json_extract_string`) without loading the whole file into memory.

## 2. How the LLM connects to the KG

The LLM (OpenAI, via `chat.py`) never touches Parquet, SQL, or DuckDB directly. It only
sees five **tool/function definitions** — a JSON schema describing what each tool takes and
returns:

| tool | what it does under the hood |
|---|---|
| `search_nodes(query, label?, limit?)` | `SELECT id, label, name FROM nodes WHERE name ILIKE '%query%'` |
| `get_node(node_id)` | full row lookup + `properties` JSON parsed |
| `get_neighbors(node_id, relation?, limit?)` | join `edges` to `nodes` on both `from`/`to` to resolve neighbor names |
| `rank_neighbors(node_id, relation?, sort_by?, top_n?)` | same join as above, plus `TRY_CAST(json_extract(properties, sort_by))` and `ORDER BY ... DESC LIMIT top_n` — for "strongest"/"top" questions |
| `get_edges_between(id_a, id_b)` | direct edges connecting two known ids |
| `find_paths(id_a, id_b, max_hops?)` | capped breadth-first search over `edges.parquet`: expands both directions per hop, caps fan-out per node (default 30) and frontier size (default 2000) to avoid blowing up on hub nodes, stops as soon as `id_b` is reached or `max_hops` is exhausted |
| `graph_stats()` | counts by node/relation type, for orientation questions |

This is OpenAI's **function calling** mechanism: the model is given these schemas alongside
the conversation, and instead of only returning text, it can return a structured request
like `{"name": "search_nodes", "arguments": {"query": "Parkinson disease", "label": "DIS"}}`.
`chat.py` intercepts that, calls the real Python function in `kg.py` against the actual
Parquet data, and feeds the JSON result back into the conversation as a `tool` message. The
LLM only ever reasons over what these functions return — it cannot see or invent graph data
outside of what a tool call surfaces.

The KG itself has no "understanding" — it's inert rows. All reasoning (which tool to call,
how to interpret the result, when to stop) is done by the LLM.

## 3. The reasoning loop, step by step

`chat.py` runs this loop for every user message:

1. Append the user's message to the conversation.
2. Call the OpenAI chat completion endpoint with the full conversation **and** the tool
   schemas.
3. The model responds with either:
   - plain text → print it, loop back to step 1 (wait for next user message), or
   - one or more **tool calls** → for each one, run the matching Python function in `kg.py`
     against the real Parquet data, and append the JSON result as a `tool` role message.
4. Go back to step 2 with the updated conversation (now including the tool results). The
   model sees the retrieved data and decides whether it has enough to answer, or needs to
   call another tool (e.g. resolve a name first, then fetch neighbors).
5. Repeat until the model responds with plain text instead of a tool call — that's the
   final answer, printed to the user.

This is a classic **agentic retrieval loop**: the LLM plans which lookups it needs, DuckDB
executes them against ground-truth data, and the LLM synthesizes a final answer from real
results rather than from what it already "knows" (which could be outdated or hallucinated).

The system prompt (`SYSTEM_PROMPT` in `chat.py`) enforces one important rule: **always
resolve a name to a node id via `search_nodes` before calling `get_node`/`get_neighbors`.**
This exists because graph ids are opaque strings the model can't guess (e.g. `MONDO_0005180`
for "Parkinson disease") — it must look them up first.

## 4. Worked example

Question: *"What genes are associated with Parkinson disease?"*

```
1. User → LLM: "what genes are associated with Parkinson disease?"

2. LLM → tool call: search_nodes(query="Parkinson disease", label="DIS")
   kg.py runs:
     SELECT id, label, name FROM nodes
     WHERE label = 'DIS' AND name ILIKE '%Parkinson disease%'
   → returns: [{"id": "MONDO_0005180", "type": "Disease", "name": "Parkinson disease"}]

3. Tool result → LLM (now has the real node id)

4. LLM → tool call: get_neighbors(node_id="MONDO_0005180", relation="ASSOCIATED_WITH")
   kg.py runs:
     SELECT e.from, e.to, e.relation, e.properties, n_from.name, n_to.name
     FROM edges e JOIN nodes n_from ON ... JOIN nodes n_to ON ...
     WHERE (e.from = 'MONDO_0005180' OR e.to = 'MONDO_0005180')
       AND e.relation = 'ASSOCIATED_WITH'
   → returns a list of {neighbor_id, neighbor_name, properties: {evidence_score, evidence_count, ...}}

5. Tool result → LLM (now has real genes + evidence scores)

6. LLM → final text answer, citing gene symbols, Ensembl ids, and evidence scores
   pulled directly from step 4's output — no invented data.
```

Every fact in the final answer (gene names, ids, scores) traces back to a specific SQL
query result from step 2 or 4 — the LLM's job in this pipeline is orchestration and
natural-language synthesis, not recalling biomedical facts from its own training data.

## 5. Summary diagram

```
 User question
      │
      ▼
 ┌─────────────┐   tool call (JSON)   ┌───────────┐   SQL over Parquet   ┌────────────────────┐
 │  OpenAI LLM │ ───────────────────▶ │  chat.py  │ ───────────────────▶ │  kg.py (DuckDB)     │
 │ (reasoning, │                      │ (dispatch)│                      │  nodes.parquet      │
 │  synthesis) │ ◀─────────────────── │           │ ◀─────────────────── │  edges.parquet      │
 └─────────────┘   tool result (JSON) └───────────┘   rows / JSON props  └────────────────────┘
      │
      ▼
 Final answer, grounded in retrieved rows
```
