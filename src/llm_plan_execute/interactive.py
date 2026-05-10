"""Typed CLI prompts separated from workflow decisions."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

from .selection_parser import parse_index_selection

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 5
_CANCEL_WORKFLOW_LABEL = "Cancel workflow"


class InteractiveCanceledError(Exception):
    """User canceled via Ctrl+C, explicit cancel choice, or EOF."""


class SupportsReadline(Protocol):
    def readline(self) -> str: ...


class SupportsWrite(Protocol):
    def write(self, text: str) -> None: ...

    def flush(self) -> None: ...


@dataclass(frozen=True)
class ChoiceOption(Generic[T]):
    key: str
    label: str
    value: T


@dataclass(frozen=True)
class PlanReviewDecision:
    type: Literal["accept", "modify", "stepThrough", "cancel"]


@dataclass(frozen=True)
class StageTransitionDecision:
    type: Literal["proceed", "pause", "cancel"]


@dataclass(frozen=True)
class BuildReviewDecision:
    type: Literal["applyAll", "select", "feedback", "continueWithoutApplying", "cancel"]
    recommendation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompletionReportDecision:
    type: Literal["terminal", "html", "both", "skip", "cancel"]


@dataclass(frozen=True)
class StepThroughOutcome:
    type: Literal["next", "stop", "cancel"]


class InteractiveSession:
    """Prompt rendering + typed decisions with retry, EOF, and non-interactive defaults."""

    def __init__(
        self,
        *,
        stdin: SupportsReadline | None = None,
        stdout: SupportsWrite | None = None,
        stderr: SupportsWrite | None = None,
        non_interactive: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._stdin: SupportsReadline = stdin or sys.stdin
        self._stdout: SupportsWrite = stdout or sys.stdout
        self._stderr: SupportsWrite = stderr or sys.stderr
        self.non_interactive = non_interactive
        self.max_retries = max_retries

    def prompt_choice(self, question: str, options: Sequence[ChoiceOption[T]]) -> T:
        self._println(question, stream=self._stdout)
        for option in options:
            self._println(f"  {option.key}. {option.label}", stream=self._stdout)
        choice_map = {option.key.strip().lower(): option.value for option in options}

        if self.non_interactive:
            default_key = options[0].key.strip().lower()
            self._println(f"[non-interactive] defaulting to {default_key}", stream=self._stderr)
            return choice_map[default_key]

        for _attempt in range(self.max_retries):
            raw = self._read_line()
            if raw is None:
                raise InteractiveCanceledError("EOF while choosing an option.")
            token = raw.strip().lower()
            if not token:
                self._println("Please enter a choice.", stream=self._stderr)
                continue
            if token in choice_map:
                return choice_map[token]
            self._println(f"Invalid choice {raw!r}. Try again.", stream=self._stderr)

        raise InteractiveCanceledError("Exceeded maximum retries for menu input.")

    def prompt_confirm(self, question: str, *, default_yes: bool = False) -> bool:
        suffix = " [Y/n]" if default_yes else " [y/N]"
        self._println(question + suffix, stream=self._stdout)
        if self.non_interactive:
            choice = default_yes
            self._println(f"[non-interactive] default confirm={choice}", stream=self._stderr)
            return choice
        for _attempt in range(self.max_retries):
            raw = self._read_line()
            if raw is None:
                raise InteractiveCanceledError("EOF while confirming.")
            token = raw.strip().lower()
            if token == "":
                return default_yes
            if token in {"y", "yes"}:
                return True
            if token in {"n", "no"}:
                return False
            self._println("Please answer y or n.", stream=self._stderr)
        raise InteractiveCanceledError("Exceeded maximum retries for confirmation.")

    def prompt_free_text(self, question: str, *, required: bool = True) -> str:
        self._println(question, stream=self._stdout)
        if self.non_interactive:
            self._println("[non-interactive] skipping free-text input.", stream=self._stderr)
            return ""
        for attempt in range(self.max_retries):
            raw = self._read_line()
            if raw is None:
                raise InteractiveCanceledError("EOF while reading text.")
            text = raw.strip()
            if text or not required:
                return text
            if attempt + 1 >= self.max_retries:
                break
            self._println("Input cannot be empty.", stream=self._stderr)
        raise InteractiveCanceledError("Exceeded maximum retries for required text.")

    def step_through_sections(self, title: str, sections: Sequence[tuple[str, str]]) -> StepThroughOutcome:
        self._println(title, stream=self._stdout)
        if self.non_interactive:
            self._println("[non-interactive] skipping step-through.", stream=self._stderr)
            return StepThroughOutcome(type="stop")

        index = 0
        while index < len(sections):
            heading, body = sections[index]
            self._println("", stream=self._stdout)
            self._println(f"## {heading}", stream=self._stdout)
            self._println(body.rstrip(), stream=self._stdout)
            self._println("", stream=self._stdout)
            self._println(
                "Section controls: [n] next  [s] stop and return to plan menu  [c] cancel workflow",
                stream=self._stdout,
            )
            raw = self._read_line()
            if raw is None:
                raise InteractiveCanceledError("EOF during step-through.")
            token = raw.strip().lower()
            if token in {"", "n", "next"}:
                index += 1
                continue
            if token in {"s", "stop"}:
                return StepThroughOutcome(type="stop")
            if token in {"c", "cancel"}:
                return StepThroughOutcome(type="cancel")
            self._println("Invalid section control; try n, s, or c.", stream=self._stderr)

        return StepThroughOutcome(type="stop")

    def ask_plan_review(self) -> PlanReviewDecision:
        options = (
            ChoiceOption("1", "Accept plan", PlanReviewDecision(type="accept")),
            ChoiceOption("2", "Modify plan", PlanReviewDecision(type="modify")),
            ChoiceOption("3", "Step through plan", PlanReviewDecision(type="stepThrough")),
            ChoiceOption("4", _CANCEL_WORKFLOW_LABEL, PlanReviewDecision(type="cancel")),
        )
        return self.prompt_choice("What would you like to do?", options)

    def ask_stage_transition(self) -> StageTransitionDecision:
        options = (
            ChoiceOption("1", "Yes, proceed", StageTransitionDecision(type="proceed")),
            ChoiceOption("2", "No, don't proceed", StageTransitionDecision(type="pause")),
            ChoiceOption("3", _CANCEL_WORKFLOW_LABEL, StageTransitionDecision(type="cancel")),
        )
        return self.prompt_choice("Proceed to the next stage?", options)

    def ask_build_review(self) -> BuildReviewDecision:
        if self.non_interactive:
            self._println("[non-interactive] continuing without applying recommendations.", stream=self._stderr)
            return BuildReviewDecision(type="continueWithoutApplying")
        options = (
            ChoiceOption("1", "Apply all changes", BuildReviewDecision(type="applyAll")),
            ChoiceOption("2", "Select changes to apply", BuildReviewDecision(type="select")),
            ChoiceOption("3", "Give feedback and rerun review", BuildReviewDecision(type="feedback")),
            ChoiceOption(
                "4",
                "Continue without applying",
                BuildReviewDecision(type="continueWithoutApplying"),
            ),
            ChoiceOption("5", _CANCEL_WORKFLOW_LABEL, BuildReviewDecision(type="cancel")),
        )
        return self.prompt_choice("Review found recommended changes. What would you like to do?", options)

    def ask_completion_report(self) -> CompletionReportDecision:
        if self.non_interactive:
            self._println("[non-interactive] skipping completion report prompts.", stream=self._stderr)
            return CompletionReportDecision(type="skip")
        options = (
            ChoiceOption("1", "Print in terminal", CompletionReportDecision(type="terminal")),
            ChoiceOption("2", "Create HTML report", CompletionReportDecision(type="html")),
            ChoiceOption("3", "Both", CompletionReportDecision(type="both")),
            ChoiceOption("4", "Skip", CompletionReportDecision(type="skip")),
            ChoiceOption("5", _CANCEL_WORKFLOW_LABEL, CompletionReportDecision(type="cancel")),
        )
        return self.prompt_choice("Workflow complete. How would you like the report?", options)

    def read_recommendation_selection(self, count: int) -> tuple[str, ...]:
        self._println(
            "Enter recommendation numbers (1-based), comma-separated. Example: 1, 3",
            stream=self._stdout,
        )
        if self.non_interactive:
            self._println("[non-interactive] applying none via selection.", stream=self._stderr)
            return ()
        raw = self._read_line()
        if raw is None:
            raise InteractiveCanceledError("EOF during recommendation selection.")
        return parse_index_selection(raw, count)

    def read_build_feedback(self) -> str:
        return self.prompt_free_text("Describe what should change in the next review pass:", required=True)

    def read_plan_feedback(self) -> str:
        return self.prompt_free_text("What should change in the plan?", required=True)

    def _read_line(self) -> str | None:
        try:
            line = self._stdin.readline()
        except KeyboardInterrupt as exc:
            raise InteractiveCanceledError("Interrupted.") from exc
        if line == "":
            return None
        return line.rstrip("\n")

    def _println(self, text: str, *, stream: SupportsWrite) -> None:
        stream.write(text + "\n")
        stream.flush()


class ListBuffer:
    """Simple writable buffer that records lines for tests."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        self.lines.append(text)

    def flush(self) -> None:
        return None


def session_with_mock_stdin(
    lines: Sequence[str], *, non_interactive: bool = False, max_retries: int = DEFAULT_MAX_RETRIES
) -> tuple[InteractiveSession, ListBuffer, ListBuffer]:
    """Return a session with scripted stdin plus stdout/stderr buffers."""

    iterator = iter(lines)

    class _Stdin:
        def readline(self) -> str:
            try:
                return next(iterator) + "\n"
            except StopIteration:
                return ""

    stdin = _Stdin()
    stdout = ListBuffer()
    stderr = ListBuffer()
    session = InteractiveSession(
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        non_interactive=non_interactive,
        max_retries=max_retries,
    )
    return session, stdout, stderr
