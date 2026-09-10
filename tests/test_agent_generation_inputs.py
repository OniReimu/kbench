import ast
from pathlib import Path

import pytest

import chcons

RELEASE_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_AGENT = RELEASE_ROOT.parent / "src/chcons/agent.py"
PACKAGE_AGENT = Path(chcons.__file__).resolve().parent / "agent.py"
MONOREPO_AGENT_CASE = (
    MONOREPO_AGENT
    if MONOREPO_AGENT.exists()
    else pytest.param(
        MONOREPO_AGENT,
        marks=pytest.mark.skip(reason=f"monorepo-only: {MONOREPO_AGENT} absent"),
    )
)


@pytest.mark.parametrize(
    "agent_path",
    (MONOREPO_AGENT_CASE, PACKAGE_AGENT),
)
def test_local_generation_disables_token_type_ids(agent_path: Path) -> None:
    """Every local generation path must request decoder-only model inputs.

    A generic ``PreTrainedTokenizerFast`` otherwise emits ``token_type_ids`` even
    for Mistral, whose causal-LM forward rejects that unused keyword.
    """
    tree = ast.parse(agent_path.read_text())
    generation_methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"elicit_summary", "_generate_block"}
    }
    assert set(generation_methods) == {"elicit_summary", "_generate_block"}

    for method_name, method in generation_methods.items():
        tokenizer_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "tokenizer"
        ]
        assert len(tokenizer_calls) == 1, (agent_path, method_name)
        keywords = {keyword.arg: keyword.value for keyword in tokenizer_calls[0].keywords}
        value = keywords.get("return_token_type_ids")
        assert isinstance(value, ast.Constant) and value.value is False, (
            agent_path,
            method_name,
        )
