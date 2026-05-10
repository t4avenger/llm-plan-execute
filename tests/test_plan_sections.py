from llm_plan_execute.plan_sections import split_plan_sections


def test_split_plan_sections_groups_level_two_headings():
    md = "## Goals\nfirst\n## Work\nsecond line\n"
    sections = split_plan_sections(md)
    assert sections[0][0] == "Goals"
    assert "first" in sections[0][1]
    assert sections[1][0] == "Work"


def test_split_plan_sections_keeps_overview_and_plain_text_fallback():
    sections = split_plan_sections("Intro\n\n## Goals\nfirst")

    assert sections[0] == ("Overview", "Intro")
    assert sections[1] == ("Goals", "first")
    assert split_plan_sections("Just do the work") == [("Overview", "Just do the work")]


def test_split_plan_sections_uses_default_for_empty_heading():
    assert split_plan_sections("## \nBody") == [("Section", "Body")]
