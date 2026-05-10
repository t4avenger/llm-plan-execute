import json

from llm_plan_execute.build_review_schema import (
    BuildRecommendation,
    expand_with_dependencies,
    map_numeric_selection_to_ids,
    parse_recommendations_from_summary,
    selection_requires_missing_dependency,
)


def test_parse_prefers_embedded_json_over_headings():
    payload = json.dumps(
        [
            {
                "id": "stable-one",
                "title": "From JSON",
                "description": "desc",
                "depends_on": [],
            }
        ]
    )
    md = f"""# Summary

<!-- llm-plan-execute:recommendations
{payload}
-->

### Heading noise

This would become heading-1 if JSON were ignored.
"""
    recs = parse_recommendations_from_summary(md)
    assert len(recs) == 1
    assert recs[0].id == "stable-one"
    assert recs[0].title == "From JSON"


def test_numeric_selection_maps_embedded_ids():
    recs = [
        BuildRecommendation(id="fix-tests", title="Tests", description="d"),
        BuildRecommendation(id="fix-docs", title="Docs", description="d"),
    ]
    assert map_numeric_selection_to_ids(("2", "1"), recs) == ("fix-docs", "fix-tests")


def test_expand_dependencies_order():
    recs = [
        BuildRecommendation(id="base", title="b", description="", depends_on=()),
        BuildRecommendation(id="top", title="t", description="", depends_on=("base",)),
    ]
    assert expand_with_dependencies(["top"], recs) == ["base", "top"]


def test_missing_dependency_id_not_in_list_reports():
    recs = [
        BuildRecommendation(id="orphan-dep", title="x", description="", depends_on=("not-a-rec",)),
    ]
    expanded = expand_with_dependencies(["orphan-dep"], recs)
    assert expanded == ["orphan-dep"]
    msg = selection_requires_missing_dependency(expanded, recs)
    assert msg is not None
    assert "not-a-rec" in msg


def test_selection_requires_missing_dependency_direct():
    recs = [
        BuildRecommendation(id="r2", title="two", description="", depends_on=("r1",)),
        BuildRecommendation(id="r1", title="one", description="", depends_on=()),
    ]
    assert selection_requires_missing_dependency(["r2"], recs) is not None
    assert selection_requires_missing_dependency(["r1", "r2"], recs) is None
