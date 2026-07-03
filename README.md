# chat_with_optimusKG

Chat interface for [OptimusKG](https://optimuskg.ai), a biomedical knowledge graph, using
DuckDB to query the graph's Parquet files and OpenAI function calling to answer natural
language questions grounded in real nodes and relationships.

## Setup

```bash
pip install -r requirements.txt
```

Download the core graph files (~326 MB) into `data/`:

```bash
mkdir -p data
curl -L -o data/nodes.parquet "https://dataverse.harvard.edu/api/access/datafile/13835035"
curl -L -o data/edges.parquet "https://dataverse.harvard.edu/api/access/datafile/13835018"
```

Create a `.env` file in this directory:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
python chat.py
```

Ask questions like "what genes are associated with Parkinson disease?" — the assistant
resolves entity names to graph node ids via `search_nodes`, then walks relationships via
`get_neighbors`/`get_edges_between`, and answers citing real node ids and relation types.

## Files

- `kg.py` — DuckDB query layer over `nodes.parquet`/`edges.parquet`
- `chat.py` — CLI chat loop wiring OpenAI tool calls to `kg.py`
