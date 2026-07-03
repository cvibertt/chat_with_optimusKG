"""Chat with OptimusKG: an OpenAI function-calling loop backed by DuckDB over the KG Parquet files.

Usage:
    # reads OPENAI_API_KEY from .env in this directory
    python chat.py
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

import kg

load_dotenv(override=True)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a research assistant with tool access to OptimusKG, a biomedical \
knowledge graph (genes, diseases, drugs, phenotypes, anatomy, biological processes, molecular \
functions, cellular components, pathways, and environmental exposures).

Use the tools to look up real nodes and relationships before answering questions about biomedical \
entities or their relationships. Always resolve a name to a node id via search_nodes before calling \
get_node or get_neighbors. Cite node ids and relation types in your answer. If a search returns \
multiple plausible matches, briefly disambiguate or ask the user which one they mean rather than \
guessing.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_nodes",
            "description": "Search KG nodes by name/symbol substring (case-insensitive). Returns id, type, name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for, e.g. a gene symbol or drug/disease name."},
                    "label": {
                        "type": "string",
                        "description": "Optional node type filter.",
                        "enum": list(kg.LABELS.keys()),
                    },
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_node",
            "description": "Get full properties for one node by its exact id.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": "Get edges/relationships touching a node (either direction), with neighbor names resolved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "relation": {"type": "string", "description": "Optional relation type filter, e.g. ASSOCIATED_WITH, INDICATION, INTERACTS_WITH."},
                    "limit": {"type": "integer", "default": 25},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_edges_between",
            "description": "Get direct edges (either direction) between two specific node ids.",
            "parameters": {
                "type": "object",
                "properties": {"id_a": {"type": "string"}, "id_b": {"type": "string"}},
                "required": ["id_a", "id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_stats",
            "description": "Get high-level counts of node types and the most common relation types in the graph.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

DISPATCH = {
    "search_nodes": lambda i: kg.search_nodes(i["query"], i.get("label"), i.get("limit", 10)),
    "get_node": lambda i: kg.get_node(i["node_id"]),
    "get_neighbors": lambda i: kg.get_neighbors(i["node_id"], i.get("relation"), i.get("limit", 25)),
    "get_edges_between": lambda i: kg.get_edges_between(i["id_a"], i["id_b"]),
    "graph_stats": lambda i: kg.graph_stats(),
}


def run_tool(name: str, tool_input: dict):
    try:
        return DISPATCH[name](tool_input)
    except Exception as e:
        return {"error": str(e)}


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found. Put it in .env in this directory.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("OptimusKG chat. Ask about genes, diseases, drugs, phenotypes, etc. Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                print(f"\nkg> {msg.content}\n")
                break

            if msg.content:
                print(f"\nkg (thinking)> {msg.content.strip()}")

            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                print(f"  [calling {call.function.name}({json.dumps(args)})]")
                result = run_tool(call.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str)[:8000],
                })


if __name__ == "__main__":
    main()
