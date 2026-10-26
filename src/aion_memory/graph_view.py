"""Render a knowledge graph export as an HTML page drawn with vis-network."""

import hashlib
import json
from collections import Counter

from aion_memory.graph_models import GraphExport, normalize_graph_name

_TYPE_PALETTE = [
    "#5eead4", "#f0abfc", "#fbbf24", "#60a5fa", "#fb7185",
    "#a3e635", "#fb923c", "#a78bfa", "#38bdf8", "#f472b6",
]  # fmt: skip

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Knowledge graph: __USER_ID__</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  body { margin: 0; background: #0f1419; color: #d7dde5;
         font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif; }
  #header { display: flex; flex-wrap: wrap; align-items: center; gap: 18px;
            padding: 10px 16px; background: #161c24; border-bottom: 1px solid #263040; }
  #header .user { font-size: 15px; font-weight: 600; color: #fff; }
  #header .count { font-size: 13px; color: #93a1b3; }
  #header label { font-size: 13px; cursor: pointer; }
  #header button { background: #263040; color: #d7dde5; border: 0; border-radius: 6px;
                   padding: 5px 12px; font: inherit; font-size: 13px; cursor: pointer; }
  #header button:hover { background: #324158; }
  #graph { height: calc(100vh - 49px); }
</style>
</head>
<body>
<div id="header">
  <span class="user">__USER_ID__</span>
  <span class="count">__ENTITY_COUNT__ entities</span>
  <span class="count">__VALID_COUNT__ valid facts</span>
  <span class="count">__INVALIDATED_COUNT__ invalidated facts</span>
  <label><input type="checkbox" id="show-invalidated"> Show invalidated facts</label>
  <button id="relayout">Re-layout</button>
</div>
<div id="graph"></div>
<script>
const graphData = __GRAPH_JSON__;
const nodes = new vis.DataSet(graphData.nodes);
const edges = new vis.DataSet(graphData.edges);
const physicsOptions = {
  solver: "forceAtlas2Based",
  forceAtlas2Based: { gravitationalConstant: -90, centralGravity: 0.008,
                      springLength: 170, springConstant: 0.06, avoidOverlap: 0.8 },
  stabilization: { iterations: 250 },
};
const network = new vis.Network(
  document.getElementById("graph"),
  { nodes, edges },
  {
    nodes: { shape: "circle", borderWidth: 2,
             shadow: { enabled: true, size: 10, color: "rgba(0,0,0,0.5)" },
             font: { color: "#0f1419", face: "Inter, Segoe UI, system-ui, sans-serif" } },
    edges: { arrows: { to: { enabled: true, scaleFactor: 0.8 } }, width: 1.5,
             color: { color: "#5b6b80", highlight: "#8fb8ff", hover: "#8fb8ff" },
             font: { size: 11, color: "#b8c3d1", strokeWidth: 0, background: "#0f1419", align: "horizontal" },
             hoverWidth: 1 },
    interaction: { hover: true, tooltipDelay: 120 },
    physics: physicsOptions,
  }
);
const relayoutStabilizeIterations = 400;
let relayoutInProgress = false;
const disablePhysics = () => network.setOptions({ physics: { enabled: false } });
network.once("stabilizationIterationsDone", disablePhysics);
const finishRelayout = () => {
  network.off("stabilized", finishRelayout);
  disablePhysics();
  relayoutInProgress = false;
};
const perturbNonHubNodes = () => {
  const span = 1600;
  nodes.update(
    graphData.nodes
      .filter((node) => !node.hub)
      .map((node) => ({
        id: node.id,
        x: (Math.random() - 0.5) * span,
        y: (Math.random() - 0.5) * span,
      }))
  );
};
document.getElementById("relayout").addEventListener("click", () => {
  if (relayoutInProgress) return;
  relayoutInProgress = true;
  network.off("stabilized", finishRelayout);
  perturbNonHubNodes();
  network.setOptions({
    physics: {
      ...physicsOptions,
      enabled: true,
      stabilization: { iterations: relayoutStabilizeIterations },
      forceAtlas2Based: {
        ...physicsOptions.forceAtlas2Based,
        randomSeed: Math.floor(Math.random() * 1e9),
      },
    },
  });
  network.on("stabilized", finishRelayout);
  network.stabilize(relayoutStabilizeIterations);
});
document.getElementById("show-invalidated").addEventListener("change", (event) => {
  edges.update(
    graphData.edges
      .filter((edge) => edge.invalidated)
      .map((edge) => ({ id: edge.id, hidden: !event.target.checked }))
  );
});
</script>
</body>
</html>
"""


def type_color(entity_type: str) -> str:
    """Return a stable palette color for an entity type."""
    index = int(hashlib.md5(entity_type.encode()).hexdigest(), 16) % len(_TYPE_PALETTE)
    return _TYPE_PALETTE[index]


def _entity_degrees_from_facts(
    fact_rows: list[dict[str, object]],
    entity_id_remap: dict[str, str],
) -> Counter[str]:
    """Incident fact count per merged entity id (valid facts only; see build_graph_data).

    Counts stored facts, not vis-network edge segments (hub spokes are ignored).
    Each fact increments source and destination once; self-loops add +2 (standard
    incident count). Endpoints are remapped through entity_id_remap after merge.
    """
    degrees: Counter[str] = Counter()

    def canonical(entity_id: object) -> str:
        """Return the merged id for an entity id."""
        key = str(entity_id)
        return entity_id_remap.get(key, key)

    for row in fact_rows:
        if row["valid_to"] is not None:
            continue
        src = canonical(row["source_entity_id"])
        dst = canonical(row["destination_entity_id"])
        degrees[src] += 1
        degrees[dst] += 1
    return degrees


# Node sizing model (entity circles only; hubs stay at fixed size in build_graph_data):
# - Degree = count of incident valid facts (valid_to IS NULL), after name-merge remap.
# - Relative scale: normalized = degree / max_degree among entities with degree > 0.
# - size = min + (max - min) * normalized^gamma so low-degree nodes stay readable and
#   high-degree hubs dominate without absurd pixel sizes in vis-network.
_MIN_ENTITY_NODE_SIZE = 14.0
_MAX_ENTITY_NODE_SIZE = 300.0
_DEGREE_SIZE_GAMMA = 2.5


def _degree_to_size(degree: int, max_degree: int) -> float:
    """Map fact degree to vis circle diameter using relative scaling (see constants above)."""
    if degree <= 0:
        return _MIN_ENTITY_NODE_SIZE
    cap = max(max_degree, 1)
    normalized = degree / cap
    span = _MAX_ENTITY_NODE_SIZE - _MIN_ENTITY_NODE_SIZE
    return _MIN_ENTITY_NODE_SIZE + span * (normalized**_DEGREE_SIZE_GAMMA)


def _font_size_for_node(size: float) -> int:
    """Baseline label font from circle size (clamped 9–26); may be reduced to fit text."""
    return int(max(9, min(26, round(10 + 0.35 * size))))


_MIN_CIRCLE_FONT = 8
_MAX_CIRCLE_FONT = 35
_CHAR_WIDTH_FACTOR = 0.55
_CIRCLE_TEXT_FILL = 0.85
_LINE_HEIGHT_FACTOR = 1.15


def _estimated_text_width(char_count: int, font_size: int) -> float:
    """Estimate the pixel width of a label from its length and font size."""
    return char_count * font_size * _CHAR_WIDTH_FACTOR


def _split_display_name_two_lines(name: str) -> tuple[str, str] | None:
    """Split a name on a space near the middle for a two-line circle label."""
    if " " not in name:
        return None
    mid = len(name) // 2
    split_at = name.rfind(" ", 0, mid + 4)
    if split_at < 0:
        split_at = name.find(" ", mid - 4)
    if split_at <= 0:
        return None
    return name[:split_at], name[split_at + 1 :]


def _single_line_fits(name: str, font_size: int, max_width: float) -> bool:
    """Return whether the name fits on one line at this font size."""
    return _estimated_text_width(len(name), font_size) <= max_width


def _two_lines_fit(
    line1: str, line2: str, font_size: int, max_width: float, max_height: float
) -> bool:
    """Return whether both lines fit the circle's width and height at this font size."""
    if _estimated_text_width(len(line1), font_size) > max_width:
        return False
    if _estimated_text_width(len(line2), font_size) > max_width:
        return False
    return 2 * font_size * _LINE_HEIGHT_FACTOR <= max_height


def _label_and_font_for_circle(name: str, size: float) -> tuple[str, dict[str, object]]:
    """Return label and vis font dict so the display name fits in the circle when possible."""
    max_extent = 2 * size * _CIRCLE_TEXT_FILL
    start_font = min(_font_size_for_node(size), _MAX_CIRCLE_FONT)

    for font_size in range(start_font, _MIN_CIRCLE_FONT - 1, -1):
        if _single_line_fits(name, font_size, max_extent):
            return name, {"size": font_size}

    split = _split_display_name_two_lines(name)
    if split is not None:
        line1, line2 = split
        for font_size in range(start_font, _MIN_CIRCLE_FONT - 1, -1):
            if _two_lines_fit(line1, line2, font_size, max_extent, max_extent):
                font: dict[str, object] = {"size": font_size, "multi": True}
                return f"{line1}\n{line2}", font

    truncated = _circle_display_label(name, size)
    font = {"size": _MIN_CIRCLE_FONT}
    if "\n" in truncated:
        font["multi"] = True
    return truncated, font


def _circle_display_label(name: str, size: float) -> str:
    """Short label for inside the circle; full name remains on node title."""
    max_chars = max(6, int(size * 0.55))
    if len(name) <= max_chars:
        return name
    if max_chars >= 14 and " " in name:
        mid = len(name) // 2
        split_at = name.rfind(" ", 0, mid + 4)
        if split_at < 0:
            split_at = name.find(" ", mid - 4)
        if split_at > 0:
            line1, line2 = name[:split_at], name[split_at + 1 :]
            per_line = max(6, (max_chars + 1) // 2)
            if len(line1) <= per_line and len(line2) <= per_line:
                return f"{line1}\n{line2}"
            if len(line1) > per_line:
                line1 = line1[: per_line - 1] + "…"
            if len(line2) > per_line:
                line2 = line2[: per_line - 1] + "…"
            return f"{line1}\n{line2}"
    if max_chars <= 1:
        return "…"
    return name[: max_chars - 1] + "…"


def _canonical_entity_id(
    entity_ids: list[str], records: dict[str, dict[str, object]]
) -> str:
    """Pick one entity_id per normalized name: oldest created_at, else lexicographic id."""

    def sort_key(entity_id: str) -> tuple[bool, object, str]:
        """Order entities by creation time, unknown times last, then by id."""
        created_at = records[entity_id].get("created_at")
        return (created_at is None, created_at or "", entity_id)

    return min(entity_ids, key=sort_key)


def _merge_entity_nodes_by_name(
    nodes: dict[str, dict],
    edges: list[dict],
    entity_records: dict[str, dict[str, object]],
) -> tuple[int, dict[str, str]]:
    """Collapse entity nodes that share a normalized name; remap edge endpoints."""
    by_name: dict[str, list[str]] = {}
    for entity_id, record in entity_records.items():
        key = normalize_graph_name(str(record["name"]))
        by_name.setdefault(key, []).append(entity_id)

    id_remap: dict[str, str] = {}
    collapsed = 0
    for entity_ids in by_name.values():
        if len(entity_ids) == 1:
            continue
        canonical_id = _canonical_entity_id(entity_ids, entity_records)
        canonical_record = entity_records[canonical_id]
        merged_types = sorted(
            {str(entity_records[eid]["entity_type"]) for eid in entity_ids}
        )
        title = str(canonical_record["entity_type"])
        if len(merged_types) > 1:
            also = ", ".join(
                t for t in merged_types if t != canonical_record["entity_type"]
            )
            title = f"{title} (also: {also})"
        alias_ids = sorted(eid for eid in entity_ids if eid != canonical_id)
        if alias_ids:
            title = f"{title}\nMerged entity_ids: {', '.join(alias_ids)}"

        canonical_node = nodes[canonical_id]
        display_name = str(canonical_record["name"])
        canonical_node["title"] = f"{display_name}\n{title}"
        canonical_node["entity_type"] = canonical_record["entity_type"]

        for entity_id in entity_ids:
            id_remap[entity_id] = canonical_id
            if entity_id != canonical_id:
                nodes.pop(entity_id, None)
                collapsed += 1

    def remap_endpoint(node_id: str) -> str:
        """Return the merged id for an edge endpoint; hub ids stay as they are."""
        if str(node_id).startswith("hub:"):
            return node_id
        return id_remap.get(node_id, node_id)

    for edge in edges:
        edge["from"] = remap_endpoint(str(edge["from"]))
        edge["to"] = remap_endpoint(str(edge["to"]))

    return collapsed, id_remap


def _fact_tooltip_lines(rows: list[dict[str, object]]) -> str:
    """Return one tooltip line per fact with its relation and validity times."""
    return "\n".join(
        f"{row['relation']} | valid_from: {row['valid_from']} | "
        f"valid_to: {row['valid_to']} | world_valid_from: "
        f"{row['world_valid_from']} | world_valid_to: {row['world_valid_to']}"
        for row in rows
    )


def build_graph_data(fact_rows: list[dict[str, object]]) -> dict[str, object]:
    """Convert fact rows into vis-network nodes and edges merged per relation and status."""
    nodes: dict[str, dict] = {}
    entity_records: dict[str, dict[str, object]] = {}

    def register_entity(row: dict[str, object], side: str) -> None:
        """Record the entity on one side of a fact row and create its node."""
        entity_id = row[f"{side}_entity_id"]
        entity_type = row[f"{side}_entity_type"]
        display_name = row[f"{side}_entity"]
        created_at = row.get(f"{side}_entity_created_at")
        entity_records[entity_id] = {
            "name": display_name,
            "entity_type": entity_type,
            "created_at": created_at,
        }
        nodes[entity_id] = {
            "id": entity_id,
            "title": f"{display_name}\n{entity_type}",
            "entity_type": entity_type,
            "shape": "circle",
            "color": {"background": type_color(entity_type), "border": "#ffffff55"},
        }

    for row in fact_rows:
        register_entity(row, "source")
        register_entity(row, "destination")

    valid_rows = [row for row in fact_rows if row["valid_to"] is None]
    invalid_rows = [row for row in fact_rows if row["valid_to"] is not None]

    by_source_relation: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in valid_rows:
        key = (row["source_entity_id"], row["relation"])
        by_source_relation.setdefault(key, []).append(row)

    fan_out_keys = {
        key
        for key, rows in by_source_relation.items()
        if len({row["destination_entity_id"] for row in rows}) >= 2
    }

    edges: list[dict] = []
    parallel_counts: Counter[frozenset[str]] = Counter()

    def add_direct_edge(
        source_id: str,
        destination_id: str,
        relation: str,
        rows: list[dict[str, object]],
        *,
        invalidated: bool,
        edge_id: str | None = None,
    ) -> None:
        """Append an edge between two entities, curving it apart from parallel edges."""
        pair = frozenset((source_id, destination_id))
        parallel_index = parallel_counts[pair]
        parallel_counts[pair] += 1
        edge: dict[str, object] = {
            "id": edge_id or rows[0]["fact_id"],
            "from": source_id,
            "to": destination_id,
            "label": f"{relation} ({len(rows)})" if len(rows) > 1 else relation,
            "title": _fact_tooltip_lines(rows),
            "fact_count": len(rows),
            "invalidated": invalidated,
            "hidden": invalidated,
            "dashes": invalidated,
            "smooth": {"type": "curvedCW", "roundness": 0.15 + 0.2 * parallel_index},
        }
        if invalidated:
            edge["color"] = {"color": "#6b7280", "hover": "#9ca3af"}
        edges.append(edge)

    for (source_id, relation), rows in by_source_relation.items():
        if (source_id, relation) not in fan_out_keys:
            by_destination: dict[str, list[dict[str, object]]] = {}
            for row in rows:
                by_destination.setdefault(row["destination_entity_id"], []).append(row)
            for destination_id, dest_rows in by_destination.items():
                add_direct_edge(
                    source_id, destination_id, relation, dest_rows, invalidated=False
                )
            continue

        hub_node_id = f"hub:{source_id}:{relation}"
        source_name = rows[0]["source_entity"]
        target_count = len({row["destination_entity_id"] for row in rows})
        nodes[hub_node_id] = {
            "id": hub_node_id,
            "label": "",
            "title": f"{source_name} {relation} ({target_count} targets)",
            "hub": True,
            "shape": "dot",
            "size": 8,
            "color": {"background": "#666666", "border": "#666666"},
        }

        pair_in = frozenset((source_id, hub_node_id))
        parallel_in = parallel_counts[pair_in]
        parallel_counts[pair_in] += 1
        fact_total = len(rows)
        edges.append(
            {
                "id": f"in:{hub_node_id}",
                "from": source_id,
                "to": hub_node_id,
                "label": f"{relation} ({fact_total})",
                "title": _fact_tooltip_lines(rows),
                "fact_count": fact_total,
                "invalidated": False,
                "hidden": False,
                "smooth": {"type": "curvedCW", "roundness": 0.15 + 0.2 * parallel_in},
            }
        )

        by_destination: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_destination.setdefault(row["destination_entity_id"], []).append(row)
        for destination_id, dest_rows in by_destination.items():
            pair_out = frozenset((hub_node_id, destination_id))
            parallel_out = parallel_counts[pair_out]
            parallel_counts[pair_out] += 1
            edges.append(
                {
                    "id": dest_rows[0]["fact_id"],
                    "from": hub_node_id,
                    "to": destination_id,
                    "label": "",
                    "title": _fact_tooltip_lines(dest_rows),
                    "fact_count": len(dest_rows),
                    "invalidated": False,
                    "hidden": False,
                    "smooth": {
                        "type": "curvedCW",
                        "roundness": 0.15 + 0.2 * parallel_out,
                    },
                }
            )

    invalid_groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in invalid_rows:
        group_key = (
            row["source_entity_id"],
            row["destination_entity_id"],
            row["relation"],
        )
        invalid_groups.setdefault(group_key, []).append(row)
    for (source_id, destination_id, relation), rows in invalid_groups.items():
        add_direct_edge(source_id, destination_id, relation, rows, invalidated=True)

    merged_collapse_count, entity_id_remap = _merge_entity_nodes_by_name(
        nodes, edges, entity_records
    )

    degrees = _entity_degrees_from_facts(fact_rows, entity_id_remap)

    entity_ids = [entity_id for entity_id, node in nodes.items() if not node.get("hub")]

    positive_degrees = [
        degrees.get(entity_id, 0)
        for entity_id in entity_ids
        if degrees.get(entity_id, 0) > 0
    ]
    max_degree = max(positive_degrees, default=1)

    for entity_id in entity_ids:
        node = nodes[entity_id]
        degree = degrees.get(entity_id, 0)
        size = _degree_to_size(degree, max_degree)
        node["size"] = size
        node["value"] = size
        full_name = str(node["title"]).split("\n", 1)[0]
        label, font = _label_and_font_for_circle(full_name, size)
        node["label"] = label
        node["font"] = font

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "valid_fact_count": len(valid_rows),
        "invalidated_fact_count": len(invalid_rows),
        "merged_collapse_count": merged_collapse_count,
        "entity_degrees": degrees,
        "max_entity_degree": max_degree,
    }


def render_html(user_id: str, graph_data: dict[str, object]) -> str:
    """Fill the HTML template with the header counts and the embedded graph JSON."""
    valid_count = int(graph_data["valid_fact_count"])
    invalidated_count = int(graph_data["invalidated_fact_count"])
    vis_payload = {"nodes": graph_data["nodes"], "edges": graph_data["edges"]}
    graph_json = json.dumps(vis_payload, default=str).replace("</", "<\\/")
    # Entity count is distinct merged display nodes (one vis node per normalized name), not raw DB rows.
    replacements = {
        "__USER_ID__": user_id.replace("&", "&amp;").replace("<", "&lt;"),
        "__ENTITY_COUNT__": str(
            sum(1 for node in graph_data["nodes"] if not node.get("hub"))
        ),
        "__VALID_COUNT__": str(valid_count),
        "__INVALIDATED_COUNT__": str(invalidated_count),
        "__GRAPH_JSON__": graph_json,
    }
    html = _HTML_TEMPLATE
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    return html


def _fact_rows(graph: GraphExport) -> list[dict[str, object]]:
    """Flatten the export into one row per fact, with the type and creation time of both endpoints."""
    entities = {entity.entity_id: entity for entity in graph.entities}
    rows: list[dict[str, object]] = []
    for fact in graph.facts:
        row: dict[str, object] = fact.model_dump()
        for side in ("source", "destination"):
            entity = entities[getattr(fact, f"{side}_entity_id")]
            row[f"{side}_entity_type"] = entity.entity_type
            row[f"{side}_entity_created_at"] = entity.created_at
        rows.append(row)
    return rows


def render_graph_html(graph: GraphExport) -> str:
    """Return a standalone HTML page that draws the graph; the page loads vis-network from unpkg."""
    return render_html(graph.user_id, build_graph_data(_fact_rows(graph)))
