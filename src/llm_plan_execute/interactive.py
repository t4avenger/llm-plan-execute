"""Typed CLI prompts separated from workflow decisions."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

from .selection_parser import parse_index_selection

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 5
_CANCEL_WORKFLOW_LABEL = "Cancel workflow"
_SLASH_HANDLED = object()  # sentinel: slash command consumed, caller should continue


def _parse_confirm_token(token: str, default_yes: bool) -> bool | None:
    """Return True/False for a recognised confirm response, None to keep prompting."""
    if not token:
        return default_yes
    if token in {"y", "yes"}:
        return True
    if token in {"n", "no"}:
        return False
    return None


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
        on_verbose_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._stdin: SupportsReadline = stdin or sys.stdin
        self._stdout: SupportsWrite = stdout or sys.stdout
        self._stderr: SupportsWrite = stderr or sys.stderr
        self.non_interactive = non_interactive
        self.max_retries = max_retries
        self._on_verbose_change = on_verbose_change
        self._ctrl_c_pressed = False

    def _print_choice_menu(self, question: str, options: Sequence[ChoiceOption[T]]) -> dict[str, T]:
        self._println(question, stream=self._stdout)
        for option in options:
            self._println(f"  {option.key}. {option.label}", stream=self._stdout)
        return {option.key.strip().lower(): option.value for option in options}

    def prompt_choice(self, question: str, options: Sequence[ChoiceOption[T]], *, include_retry: bool = False) -> T:
        choice_map = self._print_choice_menu(question, options)
        if self.non_interactive:
            raise InteractiveCanceledError("Prompt required in non-interactive mode.")
        for _attempt in range(self.max_retries):
            raw = self._read_line()
            if raw is None:
                raise InteractiveCanceledError("EOF while choosing an option.")
            token = raw.strip().lower()
            if not token:
                self._println("Please enter a choice.", stream=self._stderr)
                continue
            slash_value = self._resolve_slash_for_choice(token, options, include_retry=include_retry)
            if slash_value is _SLASH_HANDLED:
                continue
            if slash_value is not None:
                return slash_value  # type: ignore[return-value]
            if token in choice_map:
                return choice_map[token]
            self._println(f"Invalid choice {raw!r}. Try again.", stream=self._stderr)
        raise InteractiveCanceledError("Exceeded maximum retries for menu input.")

    def prompt_confirm(self, question: str, *, default_yes: bool = False) -> bool:
        suffix = " [Y/n]" if default_yes else " [y/N]"
        self._println(question + suffix, stream=self._stdout)
        if self.non_interactive:
            raise InteractiveCanceledError("Confirmation required in non-interactive mode.")
        for _attempt in range(self.max_retries):
            raw = self._read_line()
            if raw is None:
                raise InteractiveCanceledError("EOF while confirming.")
            token = raw.strip().lower()
            slash = self._handle_slash_command(token, include_retry=False)
            if slash in {"cancel", "exit"}:
                raise InteractiveCanceledError("Canceled from slash command.")
            if slash:
                continue
            result = _parse_confirm_token(token, default_yes)
            if result is not None:
                return result
            self._println("Please answer y or n.", stream=self._stderr)
        raise InteractiveCanceledError("Exceeded maximum retries for confirmation.")

    def prompt_free_text(self, question: str, *, required: bool = True) -> str:
        self._println(question, stream=self._stdout)
        if self.non_interactive:
            raise InteractiveCanceledError("Free-text input required in non-interactive mode.")
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
        return self.prompt_choice("What would you like to do?", options, include_retry=True)

    def ask_stage_transition(self) -> StageTransitionDecision:
        options = (
            ChoiceOption("1", "Yes, proceed", StageTransitionDecision(type="proceed")),
            ChoiceOption("2", "No, don't proceed", StageTransitionDecision(type="pause")),
            ChoiceOption("3", _CANCEL_WORKFLOW_LABEL, StageTransitionDecision(type="cancel")),
        )
        return self.prompt_choice("Proceed to the next stage?", options)

    def ask_build_review(self) -> BuildReviewDecision:
        if self.non_interactive:
            raise InteractiveCanceledError("Build review prompt required in non-interactive mode.")
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
        return self.prompt_choice(
            "Review found recommended changes. What would you like to do?",
            options,
            include_retry=True,
        )

    def ask_completion_report(self) -> CompletionReportDecision:
        if self.non_interactive:
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
            raise InteractiveCanceledError("Selection required in non-interactive mode.")
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
        except KeyboardInterrupt:
            if not self._ctrl_c_pressed:
                self._ctrl_c_pressed = True
                self._println("\nPress Ctrl+C again to cancel the workflow.", stream=self._stderr)
                return ""
            raise InteractiveCanceledError("Interrupted.") from None
        if line == "":
            return None
        self._ctrl_c_pressed = False
        return line.rstrip("\n")

    def _println(self, text: str, *, stream: SupportsWrite) -> None:
        stream.write(text + "\n")
        stream.flush()

    def _handle_slash_command(self, token: str, *, include_retry: bool) -> str | None:
        if not token.startswith("/"):
            return None
        if token.startswith("//"):
            self._println(f"Escaped slash input treated as data: {token[1:]}", stream=self._stderr)
            return "escaped"
        return self._dispatch_slash_command(token, include_retry=include_retry)

    def _dispatch_slash_command(self, cmd: str, *, include_retry: bool) -> str:
        if cmd == "/help":
            self._println(
                "Commands: /help, /status, /continue, /retry, /cancel, /verbose on, /verbose off, /exit",
                stream=self._stdout,
            )
            return "help"
        if cmd == "/status":
            self._println("Session is idle and awaiting next action.", stream=self._stdout)
            return "status"
        _simple = {"/continue": "continue", "/cancel": "cancel", "/exit": "exit"}
        if cmd in _simple:
            return _simple[cmd]
        if cmd == "/retry" and include_retry:
            return "retry"
        if cmd.startswith("/verbose"):
            return self._handle_verbose_command(cmd)
        self._println("Unknown slash command. Use /help.", stream=self._stderr)
        return "unknown"

    def _handle_verbose_command(self, cmd: str) -> str:
        if cmd == "/verbose on":
            return self._set_verbose(True)
        if cmd == "/verbose off":
            return self._set_verbose(False)
        self._println("Usage: /verbose on | /verbose off", stream=self._stderr)
        return "unknown"

    def _set_verbose(self, enabled: bool) -> str:
        if self._on_verbose_change:
            self._on_verbose_change(enabled)
        label = "enabled" if enabled else "disabled"
        self._println(f"Verbose mode {label} for this session.", stream=self._stdout)
        return "verbose_on" if enabled else "verbose_off"

    def _resolve_slash_for_choice(
        self,
        token: str,
        options: Sequence[ChoiceOption[T]],
        *,
        include_retry: bool,
    ) -> object:
        """Return option value, _SLASH_HANDLED sentinel, or None (not a slash command)."""
        slash = self._handle_slash_command(token, include_retry=include_retry)
        if slash in {"cancel", "exit"}:
            raise InteractiveCanceledError(f"Workflow {slash}ed.")
        if slash == "continue":
            return options[0].value
        if slash == "retry":
            return options[1].value if len(options) > 1 else options[0].value
        if slash:
            return _SLASH_HANDLED
        return None


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
