from llm_plan_execute.html_report import REPORT_HTML_NAME, deterministic_html_report_path, write_html_report
from llm_plan_execute.types import RunState


def test_html_report_path_is_stable(tmp_path):
    run = RunState.create("prompt", tmp_path)
    run.run_dir.mkdir(parents=True)
    path = write_html_report(run)
    assert path.name == REPORT_HTML_NAME
    assert path == deterministic_html_report_path(run.run_dir)
    text = path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert run.run_id in text
    assert "Content-Security-Policy" in text
    assert "<pre>" in text


def test_markdown_report_coexists_with_html(tmp_path):
    run = RunState.create("prompt", tmp_path)
    run.run_dir.mkdir(parents=True)
    md_path = run.run_dir / "report.md"
    md_path.write_text("# md\n", encoding="utf-8")
    write_html_report(run)
    assert md_path.exists()
    assert (run.run_dir / REPORT_HTML_NAME).exists()
