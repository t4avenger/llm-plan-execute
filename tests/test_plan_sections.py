from llm_plan_execute.plan_sections import split_plan_sections


def test_split_plan_sections_groups_level_two_headings():
    md = "## Goals\nfirst\n## Work\nsecond line\n"
    sections = split_plan_sections(md)
    assert sections[0][0] == "Goals"
    assert "first" in sections[0][1]
    assert sections[1][0] == "Work"
