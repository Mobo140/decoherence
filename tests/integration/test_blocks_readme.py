"""The examples of src/blocks/README.md run as written."""
from __future__ import annotations

import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parents[2] / "src" / "blocks" / (
    "README.md")
EXAMPLES = re.findall(
    r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)


def test_readme_has_an_example_per_module():
    assert len(EXAMPLES) >= 12


@pytest.mark.parametrize("index", range(len(EXAMPLES)))
def test_readme_example_runs(index, capsys):
    exec(compile(EXAMPLES[index], f"<README example {index}>", "exec"),
         {"__name__": "readme_example"})
    assert capsys.readouterr().out != ""
