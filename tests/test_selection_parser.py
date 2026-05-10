from llm_plan_execute.build_review_schema import (
    BuildRecommendation,
    expand_with_dependencies,
    map_numeric_selection_to_ids,
)
from llm_plan_execute.selection_parser import parse_index_selection


def test_parse_index_selection_trims_and_deduplicates():
    assert parse_index_selection(" 2 , 2 , 1 ", 3) == ("2", "1")


def test_parse_index_selection_drops_invalid_and_out_of_range():
    assert parse_index_selection("0, 4, x, 1-2, 2", 3) == ("2",)


def test_parse_index_selection_empty():
    assert parse_index_selection("   ", 5) == ()
    assert parse_index_selection("", 5) == ()


def test_numeric_tokens_map_to_recommendation_ids():
    recs = [
        BuildRecommendation(id="a", title="A", description="da"),
        BuildRecommendation(id="b", title="B", description="db"),
    ]
    assert map_numeric_selection_to_ids(("2", "1"), recs) == ("b", "a")


def test_dependencies_expand_deterministically():
    recs = [
        BuildRecommendation(id="r1", title="one", description="", depends_on=()),
        BuildRecommendation(id="r2", title="two", description="", depends_on=("r1",)),
    ]
    assert expand_with_dependencies(["r2"], recs) == ["r1", "r2"]
