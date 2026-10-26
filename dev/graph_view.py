"""Render one user's knowledge graph from the demo database to an HTML file (dev tool)."""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aion_memory import create_memory_system, render_graph_html
from main import load_database_url, load_openrouter_config

DEFAULT_OUTPUT_PATH = Path(__file__).parent / "data" / "graph.html"


def main() -> None:
    """Export the user's graph through the public API and write it as an HTML file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    api_key, _, extraction_model, embedding_model = load_openrouter_config()
    memory_system = create_memory_system(
        api_key, extraction_model, embedding_model, load_database_url()
    )
    try:
        graph = memory_system.export_graph(arguments.user_id)
    finally:
        memory_system.close()
    if not graph.facts:
        sys.exit(f"No facts found for user '{arguments.user_id}'.")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render_graph_html(graph), encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
