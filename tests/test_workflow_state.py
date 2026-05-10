from llm_plan_execute.workflow_state import WorkflowState


def test_terminal_report_printed_requires_literal_true():
    assert WorkflowState.from_json_dict({"terminal_report_printed": True}).terminal_report_printed is True
    assert WorkflowState.from_json_dict({"terminal_report_printed": False}).terminal_report_printed is False
    assert WorkflowState.from_json_dict({"terminal_report_printed": "false"}).terminal_report_printed is False
    assert WorkflowState.from_json_dict({}).terminal_report_printed is False
