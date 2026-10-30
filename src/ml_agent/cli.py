"""Terminal interface: interactive chat and one-shot runs, with approval prompts for code."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

from ml_agent.agent import build_agent, open_checkpointer
from ml_agent.config import EFFORT_LEVELS, Settings
from ml_agent.workspace import Workspace

TODO_MARKS = {"completed": "[green]✔[/]", "in_progress": "[yellow]▶[/]", "pending": "[dim]○[/]"}
HELP = """\
Describe what you want, e.g. "Train a model to predict `churned` in ~/data/customers.csv".

  /new      start a new conversation thread
  /thread   show the current thread id
  /help     show this help
  /exit     quit (also Ctrl+D)
"""


def _text_blocks(message: AIMessage) -> tuple[str, str]:
    """Split an AI message into visible text and (summarized) reasoning."""
    if isinstance(message.content, str):
        return message.content, ""
    text, thinking = [], []
    for block in message.content:
        if isinstance(block, str):
            text.append(block)
        elif block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "thinking" and block.get("thinking"):
            thinking.append(block["thinking"])
    return "".join(text), "\n".join(thinking)


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_args(args: dict[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        rendered = json.dumps(value, default=str) if not isinstance(value, str) else repr(value)
        parts.append(f"{key}={_shorten(rendered, 80)}")
    return ", ".join(parts)


def _summarize_result(name: str | None, data: Any) -> tuple[str | None, list[str]]:
    """One-line summary (plus notable details) of a tool result for the terminal."""
    if not isinstance(data, dict):
        return None, []
    if name == "load_dataset":
        return f"dataset '{data['name']}': {data['rows']} rows × {data['n_columns']} columns", []
    if name == "profile_dataset":
        warnings = data.get("warnings", [])
        return (
            f"{data['rows']} rows × {data['columns']} columns · {len(warnings)} warnings",
            warnings,
        )
    if name == "train_automl":
        metric = data["metric"]
        return (
            f"{data['experiment_id']} · best {data['best_estimator']} · held-out {metric} "
            f"{data['test'][metric]} (baseline {data['test']['baseline']}) · "
            f"{data['search']['trials']} trials in {data['search']['seconds']}s"
        ), [data["warning"]] if data.get("warning") else []
    if name == "explain_model":
        top = ", ".join(f"{r['feature']} ({r['importance']})" for r in data["top_features"][:5])
        return f"top features by {data['metric']} drop: {top}", []
    if name == "predict":
        return f"{data['rows_scored']} rows scored → {data['output_path']}", []
    if name == "export_model":
        return f"bundle → {data['bundle_path']}", []
    if name == "write_report":
        return f"report → {data['report_path']}", []
    return None, []


class TerminalUI:
    """Renders agent progress and collects approvals."""

    def __init__(self, console: Console, *, show_thinking: bool = True):
        self.console = console
        self.show_thinking = show_thinking
        self.approve_all = False
        self._seen: set[str] = set()
        self.status: Status | None = None

    # ------------------------------------------------------------- rendering

    def render_update(self, update: dict[str, Any]) -> None:
        for node, value in update.items():
            if node == "__interrupt__" or not isinstance(value, dict):
                continue
            if value.get("todos"):
                self._render_todos(value["todos"])
            messages = value.get("messages") or []
            for message in messages if isinstance(messages, list) else [messages]:
                if isinstance(message, BaseMessage):
                    self._render_message(message)

    def _render_message(self, message: BaseMessage) -> None:
        key = message.id or f"{message.type}:{getattr(message, 'tool_call_id', id(message))}"
        if key in self._seen:
            return
        self._seen.add(key)
        if isinstance(message, AIMessage):
            text, thinking = _text_blocks(message)
            if thinking and self.show_thinking:
                self.console.print(f"[dim italic]💭 {escape(_shorten(thinking, 400))}[/]")
            if text.strip():
                self.console.print(Markdown(text))
            for call in message.tool_calls:
                self._render_tool_call(call)
        elif isinstance(message, ToolMessage):
            self._render_tool_result(message)

    def _render_tool_call(self, call: dict[str, Any]) -> None:
        name, args = call["name"], call["args"]
        if name == "write_todos":
            return
        if name == "run_python" and self.approve_all:
            self.console.print(Panel(Syntax(args.get("code", ""), "python"), title="run_python"))
        elif name != "run_python":
            self.console.print(f"[cyan]▸ {name}[/]([dim]{escape(_format_args(args))}[/])")
        if self.status:
            label = {
                "train_automl": f"Running AutoML search ({args.get('time_budget', 60)}s budget)…",
                "explain_model": "Computing permutation importance…",
                "run_python": "Running Python…",
            }.get(name, f"Running {name}…")
            self.status.update(label)

    def _render_tool_result(self, message: ToolMessage) -> None:
        if self.status:
            # The tool has finished, so the model is working again.
            self.status.update("Thinking…")
        if message.name == "write_todos":
            return
        content = message.content if isinstance(message.content, str) else str(message.content)
        if message.status == "error":
            self.console.print(f"  [red]✗ {escape(_shorten(content, 500))}[/]")
            return
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            data = None
        if message.name == "run_python" and isinstance(data, dict):
            problem = data.get("error") or (data.get("stderr") if data.get("exit_code") else None)
            output = _shorten("\n".join(filter(None, [data.get("stdout"), problem])), 1500)
            style = "red" if problem else "dim"
            self.console.print(Padding(Text(output, style=style), (0, 0, 0, 4)))
            return
        try:
            summary, details = _summarize_result(message.name, data)
        except (KeyError, TypeError, IndexError):
            summary, details = None, []
        self.console.print(
            f"  [dim]↳ {escape(summary or _shorten(' '.join(content.split()), 200))}[/]"
        )
        for line in details:
            self.console.print(f"    [yellow]⚠ {escape(line)}[/]")

    def _render_todos(self, todos: list[dict[str, Any]]) -> None:
        lines = [
            f"{TODO_MARKS.get(t.get('status'), '○')} {escape(t.get('content', ''))}" for t in todos
        ]
        self.console.print(Panel("\n".join(lines), title="Plan", border_style="blue", expand=False))

    # -------------------------------------------------------------- approval

    def decide(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """Ask the user about each action in a human-in-the-loop request."""
        decisions = []
        for action in request["action_requests"]:
            code = action["args"].get("code", "")
            self.console.print(
                Panel(
                    Syntax(code, "python", line_numbers=True),
                    title=f"{action['name']} — approval needed",
                    subtitle=action.get("description"),
                    border_style="yellow",
                )
            )
            if self.approve_all:
                decisions.append({"type": "approve"})
                continue
            if not sys.stdin.isatty():
                decisions.append(
                    {
                        "type": "reject",
                        "message": "No one is available to approve code in this non-interactive "
                        "run. Continue without run_python, or ask the user to rerun with "
                        "--auto-approve.",
                    }
                )
                continue
            choice = Prompt.ask(
                "Run it? [b]y[/b]es, [b]n[/b]o, [b]e[/b]dit, [b]a[/b]lways",
                choices=["y", "n", "e", "a"],
                default="y",
                console=self.console,
            )
            if choice == "a":
                self.approve_all = True
                choice = "y"
            if choice == "y":
                decisions.append({"type": "approve"})
            elif choice == "e":
                edited = self._edit(code)
                if edited is None or edited == code:
                    decisions.append({"type": "approve"})
                else:
                    decisions.append(
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": action["name"],
                                "args": {**action["args"], "code": edited},
                            },
                        }
                    )
            else:
                reason = Prompt.ask(
                    "Why not? (optional, tells the agent what to do instead)",
                    default="",
                    console=self.console,
                )
                decision: dict[str, Any] = {"type": "reject"}
                if reason.strip():
                    decision["message"] = reason.strip()
                decisions.append(decision)
        return decisions

    def _edit(self, code: str) -> str | None:
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        editor = editor or next(
            (e for e in ("nano", "vim", "vi", "notepad") if shutil.which(e)), None
        )
        if editor is None:
            self.console.print("[red]No editor found; set $EDITOR. Running the original code.[/]")
            return None
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(code)
            path = handle.name
        try:
            subprocess.run([*shlex.split(editor), path], check=False)
            with open(path) as handle:
                return handle.read()
        finally:
            os.unlink(path)


def repair_dangling_tool_calls(agent: Any, config: dict[str, Any]) -> None:
    """After an interruption, answer any tool calls left without results so the thread stays valid."""
    messages = agent.get_state(config).values.get("messages", [])
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return
    missing = [call for call in last_ai.tool_calls if call["id"] not in answered]
    if missing:
        agent.update_state(
            config,
            {
                "messages": [
                    ToolMessage(
                        content="Cancelled by the user before it finished.",
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                    for call in missing
                ]
            },
            as_node="tools",
        )


def run_turn(agent: Any, config: dict[str, Any], payload: Any, ui: TerminalUI) -> None:
    """Stream one user turn to completion, pausing for approvals when the agent asks."""
    while payload is not None:
        pending = None
        with ui.console.status("Thinking…", spinner="dots") as status:
            ui.status = status
            try:
                for update in agent.stream(payload, config, stream_mode="updates"):
                    if "__interrupt__" in update:
                        pending = update["__interrupt__"][0]
                        continue
                    ui.render_update(update)
            finally:
                ui.status = None
        payload = Command(resume={"decisions": ui.decide(pending.value)}) if pending else None


def resume_pending(agent: Any, config: dict[str, Any], ui: TerminalUI) -> None:
    """If a resumed thread stopped at an approval prompt, ask again before continuing."""
    state = agent.get_state(config)
    if state.interrupts:
        ui.console.print("[yellow]This thread was waiting for your approval.[/]")
        run_turn(
            agent, config, Command(resume={"decisions": ui.decide(state.interrupts[0].value)}), ui
        )


def safe_turn(agent: Any, config: dict[str, Any], payload: Any, ui: TerminalUI) -> None:
    try:
        run_turn(agent, config, payload, ui)
    except KeyboardInterrupt:
        repair_dangling_tool_calls(agent, config)
        ui.console.print("\n[yellow]Interrupted. The conversation is kept; send a new message.[/]")


def _new_thread_id() -> str:
    return datetime.now().strftime("thread-%Y%m%d-%H%M%S")


def _check_credentials(settings: Settings, console: Console) -> bool:
    if settings.provider == "anthropic" and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        console.print(
            "[red]ANTHROPIC_API_KEY is not set.[/] Put it in a .env file or your environment, "
            "or run a local model instead, e.g. [bold]--model ollama:qwen3:14b[/]."
        )
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    # Options are accepted before or after the subcommand. SUPPRESS keeps an option that is absent
    # from the subcommand from overwriting one given before it.
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument(
        "--model", help="LLM as provider:model (default: anthropic:claude-opus-5-5)"
    )
    common.add_argument(
        "--effort", choices=EFFORT_LEVELS, help="Reasoning effort for Claude models"
    )
    common.add_argument("--workspace", help="Workspace directory (default: ./ml_workspace)")
    common.add_argument("--thread", help="Conversation thread id to resume")
    common.add_argument("--max-time-budget", type=int, help="Cap in seconds for each AutoML search")
    common.add_argument(
        "--auto-approve",
        action="store_const",
        const=True,
        help="Run agent-written Python without asking first",
    )
    common.add_argument(
        "--hide-thinking", action="store_true", help="Don't print reasoning summaries"
    )

    parser = argparse.ArgumentParser(
        prog="ml-agent",
        description="A local machine-learning engineer agent (LangChain + LangGraph + FLAML AutoML).",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("chat", parents=[common], help="Interactive session (default)")
    run = sub.add_parser("run", parents=[common], help="Run a single task and exit")
    run.add_argument("task", help="What the agent should do")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = vars(_build_parser().parse_args(argv))
    console = Console()
    settings = Settings.from_env(
        model=args.get("model"),
        effort=args.get("effort"),
        workspace=args.get("workspace"),
        max_time_budget=args.get("max_time_budget"),
        auto_approve=args.get("auto_approve"),
    )
    if not _check_credentials(settings, console):
        return 1

    ws = Workspace(settings.workspace)
    agent, _ = build_agent(settings, checkpointer=open_checkpointer(ws))
    ui = TerminalUI(console, show_thinking=not args.get("hide_thinking"))
    ui.approve_all = settings.auto_approve
    thread_id = args.get("thread") or _new_thread_id()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    console.print(
        Panel(
            f"[bold]ML engineer agent[/] · model [cyan]{settings.model}[/]"
            + (f" (effort {settings.effort})" if settings.provider == "anthropic" else "")
            + f"\nworkspace [cyan]{ws.root}[/] · thread [cyan]{thread_id}[/]"
            + (
                "\n[yellow]auto-approve is on: agent code runs without review[/]"
                if settings.auto_approve
                else ""
            ),
            border_style="green",
            expand=False,
        )
    )
    resume_pending(agent, config, ui)

    if args.get("command") == "run":
        safe_turn(agent, config, {"messages": [{"role": "user", "content": args["task"]}]}, ui)
        console.print(
            f"[dim]Continue this conversation with: ml-agent chat --thread {thread_id}[/]"
        )
        return 0

    console.print(HELP, style="dim")
    while True:
        try:
            text = Prompt.ask("[bold green]you[/]", console=console).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/help":
            console.print(HELP, style="dim")
            continue
        if text == "/thread":
            console.print(f"thread: {thread_id}")
            continue
        if text == "/new":
            thread_id = _new_thread_id()
            config["configurable"]["thread_id"] = thread_id
            console.print(f"[dim]Started thread {thread_id}[/]")
            continue
        safe_turn(agent, config, {"messages": [{"role": "user", "content": text}]}, ui)
    console.print(f"[dim]Resume with: ml-agent chat --thread {thread_id}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
