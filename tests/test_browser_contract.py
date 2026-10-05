"""Keep the explicit browser expectation in step with the public registry."""
import ast
from pathlib import Path

from global_weather.connectors import CATALOG


def test_browser_source_expectation_matches_registry():
    path = Path(__file__).resolve().parents[1] / "scripts" / "browser-test.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "EXPECTED_SOURCES"
                           for target in node.targets)]
    assert len(assignments) == 1
    expected = ast.literal_eval(assignments[0].value)
    actual = [item["id"] for item in CATALOG]
    assert set(actual) == expected
    assert len(actual) == len(expected)
