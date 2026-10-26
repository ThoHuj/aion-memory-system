"""Rendering a graph export as an HTML page."""

import json
import re

from aion_memory import GraphExport, render_graph_html
from aion_memory.graph_models import GraphEntity, StoredFact


def graph_entity(entity_id: str, name: str) -> GraphEntity:
    """Build a graph entity created at a fixed time."""
    return GraphEntity(
        entity_id=entity_id,
        name=name,
        entity_type="thing",
        created_at="2026-01-01T00:00:00+00:00",
    )


def stored_fact(
    fact_id: str, source: GraphEntity, destination: GraphEntity, valid_to: str | None
) -> StoredFact:
    """Build a stored fact between two graph entities."""
    return StoredFact(
        fact_id=fact_id,
        user_id="alice",
        source_entity_id=source.entity_id,
        source_entity=source.name,
        relation="owns",
        destination_entity_id=destination.entity_id,
        destination_entity=destination.name,
        valid_from="2026-01-01T00:00:00+00:00",
        valid_to=valid_to,
        world_valid_from="2026-01-01T00:00:00+00:00",
        world_valid_to=None,
        add_id="add",
    )


def embedded_graph(html: str) -> dict:
    """Return the nodes and edges the page embeds for vis-network."""
    match = re.search(r"const graphData = (.*);\n", html)
    assert match is not None
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_the_page_shows_counts_and_embeds_every_entity_and_fact() -> None:
    """Valid and ended facts are counted, and every entity and fact becomes a node or edge."""
    alice = graph_entity("e1", "alice")
    rex = graph_entity("e2", "rex")
    tom = graph_entity("e3", "tom")
    graph = GraphExport(
        user_id="alice",
        entities=[alice, rex, tom],
        facts=[
            stored_fact("f1", alice, rex, valid_to="2026-01-02T00:00:00+00:00"),
            stored_fact("f2", alice, tom, valid_to=None),
        ],
    )

    html = render_graph_html(graph)
    embedded = embedded_graph(html)

    assert "3 entities" in html
    assert "1 valid facts" in html
    assert "1 invalidated facts" in html
    assert {node["id"] for node in embedded["nodes"]} == {"e1", "e2", "e3"}
    edges = {edge["id"]: edge for edge in embedded["edges"]}
    assert edges["f1"]["invalidated"] is True
    assert edges["f2"]["invalidated"] is False


def test_names_cannot_break_out_of_the_page() -> None:
    """A user id or entity name containing HTML cannot close the script tag or add markup."""
    attacker = graph_entity("e1", "</script><script>alert(1)</script>")
    target = graph_entity("e2", "rex")
    graph = GraphExport(
        user_id="<b>alice</b>",
        entities=[attacker, target],
        facts=[stored_fact("f1", attacker, target, valid_to=None)],
    )

    html = render_graph_html(graph)

    # The template closes its two script tags; the name must not add a third.
    assert html.count("</script>") == 2
    assert "<b>alice" not in html


def test_the_page_loads_a_pinned_vis_network() -> None:
    """The page loads one exact vis-network version, so a new release cannot change it."""
    html = render_graph_html(GraphExport(user_id="alice"))

    assert re.search(r"unpkg\.com/vis-network@\d+\.\d+\.\d+/", html)
