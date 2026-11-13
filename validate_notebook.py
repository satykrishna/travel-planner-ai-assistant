"""Validate notebook structure and Python syntax without running any live calls."""

import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager


def main():
    path = Path(__file__).with_name("Multi_Agent_Travel_Planner.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    transformer = TransformerManager()
    definitions = set()
    secret_pattern = re.compile(r"(?:sk-proj-|sk-)[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}")
    code_cells = 0
    for index, cell in enumerate(notebook.cells, start=1):
        assert not secret_pattern.search(cell.source), f"Potential credential in cell {index}"
        assert "/Users/satya/" not in cell.source, f"Machine-specific path in cell {index}"
        if cell.cell_type != "code":
            continue
        code_cells += 1
        assert not cell.get("outputs"), f"Saved outputs in cell {index}"
        assert cell.get("execution_count") is None, f"Saved execution count in cell {index}"
        tree = ast.parse(transformer.transform_cell(cell.source), filename=f"cell-{index}")
        definitions.update(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    required = {
        "search_agent_node", "search_tool_node", "travel_scout_agent_node",
        "travel_scout_tool_node", "synthesizer_node", "dispatch_travel_tasks",
        "check_flight_hotel_dates", "classify_user_query_node",
    }
    assert required <= definitions, f"Missing definitions: {required - definitions}"
    print(f"Validated {len(notebook.cells)} cells ({code_cells} code cells); no cells executed.")


if __name__ == "__main__":
    main()
