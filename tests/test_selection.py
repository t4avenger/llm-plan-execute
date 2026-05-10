import pytest

from llm_plan_execute.selection import assign_models
from llm_plan_execute.types import ModelInfo


def test_assign_models_prefers_role_specific_models():
    models = [
        ModelInfo("a", "planner", ("planner",), 5, 2, 4, 5),
        ModelInfo("b", "builder", ("builder",), 3, 5, 1, 4),
        ModelInfo("c", "reviewer", ("plan_reviewer_a", "build_reviewer_a"), 4, 4, 2, 4),
    ]

    assignments, warnings = assign_models(models)

    assert assignments["planner"].model.id == "a:planner"
    assert assignments["builder"].model.id == "b:builder"
    assert warnings


def test_assign_models_requires_models():
    with pytest.raises(ValueError, match="No models"):
        assign_models([])
