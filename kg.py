"""Query helpers over the OptimusKG nodes/edges Parquet files, via DuckDB."""

import duckdb
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
NODES = str(DATA_DIR / "nodes.parquet")
EDGES = str(DATA_DIR / "edges.parquet")

_con = duckdb.connect()

# Node "display name" lives under different JSON keys depending on entity type
# (genes use "symbol", most others use "name"). This view normalizes that.
_con.execute(f"""
    CREATE OR REPLACE VIEW nodes AS
    SELECT
        id,
        label,
        COALESCE(
            json_extract_string(properties, '$.symbol'),
            json_extract_string(properties, '$.name')
        ) AS name,
        properties
    FROM '{NODES}'
""")

LABELS = {
    "GEN": "Gene",
    "DIS": "Disease",
    "DRG": "Drug",
    "PHE": "Phenotype",
    "ANA": "Anatomy",
    "BPO": "Biological Process",
    "MFN": "Molecular Function",
    "CCO": "Cellular Component",
    "PWY": "Pathway",
    "EXP": "Exposure",
}


def search_nodes(query: str, label: str | None = None, limit: int = 10) -> list[dict]:
    """Search nodes by name/symbol substring (case-insensitive)."""
    sql = "SELECT id, label, name FROM nodes WHERE name ILIKE ?"
    params = [f"%{query}%"]
    if label:
        sql += " AND label = ?"
        params.append(label)
    sql += " LIMIT ?"
    params.append(limit)
    rows = _con.execute(sql, params).fetchall()
    return [{"id": r[0], "label": r[1], "type": LABELS.get(r[1], r[1]), "name": r[2]} for r in rows]


def get_node(node_id: str) -> dict:
    """Fetch full properties for a node by its id."""
    row = _con.execute(
        "SELECT id, label, name, properties FROM nodes WHERE id = ?", [node_id]
    ).fetchone()
    if not row:
        return {"error": f"no node with id {node_id}"}
    return {
        "id": row[0],
        "label": row[1],
        "type": LABELS.get(row[1], row[1]),
        "name": row[2],
        "properties": json.loads(row[3]),
    }


def get_neighbors(node_id: str, relation: str | None = None, limit: int = 25) -> list[dict]:
    """Get edges touching a node (in either direction), with neighbor names resolved."""
    sql = """
        SELECT e."from", e."to", e.relation, e.label, e.properties,
               n_from.name AS from_name, n_to.name AS to_name
        FROM '{edges}' e
        JOIN nodes n_from ON n_from.id = e."from"
        JOIN nodes n_to ON n_to.id = e."to"
        WHERE e."from" = ? OR e."to" = ?
    """.format(edges=EDGES)
    params = [node_id, node_id]
    if relation:
        sql += " AND e.relation = ?"
        params.append(relation)
    sql += " LIMIT ?"
    params.append(limit)
    rows = _con.execute(sql, params).fetchall()
    out = []
    for f, t, rel, lbl, props, fname, tname in rows:
        other_id, other_name = (t, tname) if f == node_id else (f, fname)
        out.append({
            "relation": rel,
            "edge_type": lbl,
            "neighbor_id": other_id,
            "neighbor_name": other_name,
            "properties": json.loads(props) if props else {},
        })
    return out


def get_edges_between(id_a: str, id_b: str) -> list[dict]:
    """Get direct edges (either direction) between two specific node ids."""
    rows = _con.execute(
        f"""SELECT "from", "to", relation, label, properties FROM '{EDGES}'
            WHERE ("from" = ? AND "to" = ?) OR ("from" = ? AND "to" = ?)""",
        [id_a, id_b, id_b, id_a],
    ).fetchall()
    return [
        {"from": f, "to": t, "relation": rel, "edge_type": lbl, "properties": json.loads(p) if p else {}}
        for f, t, rel, lbl, p in rows
    ]


def graph_stats() -> dict:
    """High-level counts of node types and relation types."""
    node_counts = _con.execute(
        "SELECT label, count(*) FROM nodes GROUP BY label ORDER BY 2 DESC"
    ).fetchall()
    edge_counts = _con.execute(
        f"SELECT relation, count(*) FROM '{EDGES}' GROUP BY relation ORDER BY 2 DESC LIMIT 30"
    ).fetchall()
    return {
        "node_counts": {LABELS.get(l, l): c for l, c in node_counts},
        "top_relations": {r: c for r, c in edge_counts},
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(search_nodes("cetirizine"))
    pprint.pprint(graph_stats())
