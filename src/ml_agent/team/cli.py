"""Terminal front end for ``ml-agent team``: stage banners, plan approval, results table."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.types import Command
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from ml_agent.agent import open_checkpointer
from ml_agent.cli import TerminalUI
from ml_agent.config import Settings
from ml_agent.team import evaluation
from ml_agent.team.graph import TeamOptions, build_team
from ml_agent.team.runs import RunStore
from ml_agent.workspace import Workspace

AGENT_TITLES = {
    "planner": "Planner agent · understand & plan",
    "data": "Data agent · access & preparation",
    "analysis": "Analysis agent · explore & analyze",
    "modeling": "Modeling agent · model & predict",
    "review": "Review agent · quality gate",
    "signoff": "Review agent · sign-off",
    "visualization": "Visualization agent · communicate & act",
}
STATUS = {
    "tune_families": "Tuning model families on the tuning folds…",
    "run_flaml": "FLAML AutoML is searching…",
    "build_ensemble": "Scoring the ensemble on the selection folds…",
    "tune_threshold": "Tuning the decision threshold…",
    "check_leakage": "Scoring each feature on its own…",
    "rescore_candidate": "Re-running cross-validation…",
    "quick_baselines": "Cross-validating quick baselines…",
    "feature_associations": "Measuring feature-target associations…",
    "define_features": "Checking the feature function…",
    "make_chart": "Rendering chart…",
}


# Structured-report "tools" (used when a model has no native structured output). The stage
# summaries show the reports, so these calls are not rendered.
REPORT_TOOLS = {
    "ProblemSpec",
    "DataReport",
    "AnalysisReport",
    "ModelingReport",
    "ReviewVerdict",
    "SignOff",
    "VisualReport",
}


def num(value: Any) -> str:
    """Compact numbers for the terminal: 0.7945, 12.35, 1,234."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3f}" if abs(value) >= 1 else f"{value:.4f}"


def _cv_score(row: dict[str, Any]) -> Any:
    return next(v for k, v in row.items() if k.startswith("cv_") and k != "cv_std")


class TeamUI(TerminalUI):
    """TerminalUI plus agent banners, plan approval and team-specific result summaries."""

    def __init__(self, console: Console, *, show_thinking: bool = True):
        super().__init__(console, show_thinking=show_thinking)
        self.current_agent: str | None = None

    def agent_update(self, namespace: tuple[str, ...], update: dict[str, Any]) -> None:
        role = namespace[0].split(":")[0]
        if role != self.current_agent:
            self.current_agent = role
            self.console.print(Rule(f"[bold]{AGENT_TITLES.get(role, role)}[/]", style="blue"))
            if self.status:
                self.status.update(f"{AGENT_TITLES.get(role, role).split(' · ')[0]} is working…")
        self.render_update(update)

    def _render_message(self, message: BaseMessage) -> None:
        # A structured report arrives as a JSON message; the stage summary shows it instead, so
        # keep only the reasoning summary.
        text = message.text.strip() if isinstance(message, AIMessage) else ""
        if not getattr(message, "tool_calls", None) and text.startswith("{") and text.endswith("}"):
            content = message.content if isinstance(message.content, list) else []
            thinking = [b for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
            message = message.model_copy(update={"content": thinking})
        super()._render_message(message)

    def _render_tool_call(self, call: dict[str, Any]) -> None:
        if call["name"] not in REPORT_TOOLS:
            super()._render_tool_call(call)

    def _render_tool_result(self, message) -> None:
        if message.name not in REPORT_TOOLS:
            super()._render_tool_result(message)

    def status_label(self, name: str, args: dict[str, Any]) -> str:
        return STATUS.get(name) or super().status_label(name, args)

    def summarize(self, name: str | None, data: Any) -> tuple[str | None, list[str]]:
        if not isinstance(data, dict):
            return super().summarize(name, data)
        if name == "tune_families":
            parts = [
                f"{c['id']} {num(_cv_score(c))} ± {num(c['cv_std'])}" for c in data["candidates"]
            ]
            errors = [f"{k}: {v}" for k, v in (data.get("errors") or {}).items()]
            budget = data["budget_remaining_seconds"]
            return f"{'; '.join(parts) or 'no candidates'} · budget left {budget}s", errors
        if name in ("run_flaml", "build_ensemble"):
            budget = data["budget_remaining_seconds"]
            return (
                f"{data['id']} {num(_cv_score(data))} ± {num(data['cv_std'])} · budget left {budget}s",
                [],
            )
        if name == "leaderboard":
            metric = data["metric"]
            top = ", ".join(f"{r['id']} {num(r[f'cv_{metric}'])}" for r in data["candidates"][:3])
            return f"top: {top or 'none yet'}", []
        if name == "feature_associations":
            top = ", ".join(
                f"{r['feature']} ({num(r['mutual_information'])})" for r in data["top_features"][:5]
            )
            return f"most informative: {top}", []
        if name == "rescore_candidate":
            ok = "reproducible" if data["reproducible"] else "NOT reproducible"
            return (
                f"{data['candidate']}: {ok} (max fold difference {data['max_fold_difference']})",
                [],
            )
        if name == "check_overfitting":
            flagged = [r["id"] for r in data["candidates"] if r["flag"]]
            return f"large train/CV gap: {', '.join(flagged) or 'none'}", []
        if name == "check_success":
            beats = data.get("beats_dummy")
            return (
                f"{data['candidate']}: {'beats' if beats else 'does not beat'} the dummy baseline",
                [],
            )
        if name == "list_model_families":
            usable = [f["family"] for f in data["families"] if not f["unavailable"]]
            return f"usable: {', '.join(usable)} · budget {data['budget_remaining_seconds']}s", []
        if name in ("get_column_roles", "set_column_roles"):
            features = data.get("features") or [
                *data.get("numeric", []),
                *data.get("categorical", []),
                *data.get("high_cardinality", []),
            ]
            return f"{len(features)} features: {', '.join(features)}", []
        if name == "profile_train":
            return super().summarize("profile_dataset", data)
        if name == "compare_candidates":
            return f"best {data['best']} · 1-SE pick {data['one_se_recommendation']}", []
        if name == "check_leakage":
            flagged = data["suspicious"]
            return ("no single feature explains the target" if not flagged else None), (
                [f"possible leakage: {', '.join(flagged)}"] if flagged else []
            )
        if name == "define_features":
            return f"features added: {', '.join(data['columns_added']) or 'none'}", []
        if name == "tune_threshold":
            return f"threshold {data['threshold']} for {data['objective']}", []
        if name in ("make_chart", "write_report"):
            return f"saved → {data.get('path') or data.get('report_path')}", []
        if name == "quick_baselines":
            metric = data["metric"]
            return " · ".join(f"{r['model']}: {r[metric]}" for r in data["results"]), []
        return super().summarize(name, data)

    # --------------------------------------------------------------- stages

    def stage_update(self, node: str, update: dict[str, Any] | None) -> None:
        """Summaries of the parent graph's steps."""
        if not update:
            return
        out = self.console.print
        if node == "make_splits":
            s = update["splits"]
            out(
                f"[green]✔[/] Locked {s['rows_test']} test rows away; {s['rows_train']} training rows · "
                f"selection CV {s['selection_cv']}, tuning CV {s['tuning_cv']}"
            )
        elif node in ("data", "analysis", "modeling"):
            report = update.get(f"{node}_report") or {}
            out(
                Panel(
                    escape(report.get("summary", "")),
                    title=f"{node.title()} report",
                    border_style="green",
                    expand=False,
                )
            )
            if node == "modeling":
                out(f"  proposed champion: [bold]{report.get('proposed_champion')}[/]")
        elif node == "review":
            verdict = update.get("verdict") or {}
            if verdict.get("decision") == "approve":
                out(
                    f"[green]✔ Review approved {verdict.get('champion')}[/] — {escape(verdict.get('summary', ''))}"
                )
            else:
                out(
                    f"[yellow]↺ Review asks for changes ({verdict.get('revisit_stage')})[/]: {escape(verdict.get('summary', ''))}"
                )
                for change in verdict.get("requested_changes", []):
                    out(f"    • {escape(change)}")
            if update.get("unresolved_issues"):
                out(
                    "[yellow]Out of review rounds — evaluating the best available model with open issues.[/]"
                )
        elif node == "final_evaluation":
            final = update["final"]
            metric = final["metric"]
            out(
                f"[green]✔[/] Test set (used once): {final['champion']} {metric} "
                f"[bold]{final['test'][metric]}[/] vs baseline {final['test']['baseline']}"
            )
        elif node == "signoff":
            signoff = update["signoff"]
            mark = (
                "[green]✔ trustworthy[/]" if signoff["trustworthy"] else "[red]✗ not trustworthy[/]"
            )
            out(f"{mark} — {escape(signoff['summary'])}")

    def confirm_plan(self, value: dict[str, Any]) -> dict[str, Any]:
        spec = value["spec"]
        rows = [
            ("Dataset → target", f"{spec['dataset']} → {spec['target']} ({spec['task']})"),
            ("Primary metric", spec["primary_metric"]),
            ("Also report", ", ".join(spec["secondary_metrics"]) or "—"),
            ("Positive class", spec.get("positive_class") or "—"),
            ("Success criteria", spec["success_criteria"]),
            ("Excluded columns", ", ".join(spec["excluded_columns"]) or "—"),
            ("Groups kept together", spec.get("group_column") or "—"),
            (
                "Validation",
                f"{spec['cv_folds']}-fold CV × 2 repeats; {spec['test_size']:.0%} locked test set",
            ),
            ("Threshold objective", spec.get("threshold_objective") or "—"),
            (
                "Budget",
                f"{value['budget_minutes']:g} min compute, ≤ {value['max_iterations']} review rounds",
            ),
        ]
        rows += [("Assumption", a) for a in spec.get("assumptions", [])]
        rows += [("Constraint", c) for c in spec.get("constraints", [])]
        table = Table.grid(padding=(0, 2))
        for key, val in rows:
            table.add_row(f"[dim]{key}[/]", escape(str(val)))
        self.console.print(Panel(table, title="Proposed plan", border_style="yellow", expand=False))
        if not sys.stdin.isatty():
            self.console.print("[dim]Non-interactive session: plan accepted.[/]")
            return {"approve": True}
        answer = Prompt.ask(
            "Approve the plan? [b]y[/b] to start, or describe what to change",
            default="y",
            console=self.console,
        ).strip()
        return {"approve": True} if answer.lower() in ("y", "yes", "") else {"feedback": answer}


def stream_team(graph: Any, config: dict[str, Any], payload: Any, ui: TeamUI) -> None:
    """Run the team to completion, answering plan and code approvals.

    ``payload`` is the initial input, a resume ``Command``, or ``None`` to continue from the last
    checkpoint (after a crash or Ctrl+C).
    """
    while True:
        pending = None
        with ui.console.status("Starting…", spinner="dots") as status:
            ui.status = status
            try:
                for namespace, update in graph.stream(
                    payload, config, stream_mode="updates", subgraphs=True
                ):
                    if "__interrupt__" in update:
                        if not namespace:  # sub-agent interrupts are also reported at the root
                            pending = update["__interrupt__"][0]
                        continue
                    if namespace:
                        ui.agent_update(namespace, update)
                    else:
                        for node, node_update in update.items():
                            ui.stage_update(node, node_update)
            finally:
                ui.status = None
        if pending is None:
            return
        if pending.value.get("type") == "confirm_plan":
            payload = Command(resume=ui.confirm_plan(pending.value))
        else:
            payload = Command(resume={"decisions": ui.decide(pending.value)})


def print_results(console: Console, ws: Workspace, state: dict[str, Any]) -> None:
    run = RunStore.open(ws, state["run_id"])
    board = evaluation.leaderboard(run)
    metric = board["metric"]
    champion = state.get("champion")
    table = Table(
        title=f"Leaderboard · {metric} on identical selection folds", title_justify="left"
    )
    for column in ("#", "candidate", f"{metric} (mean ± std)", f"train {metric}", "complexity"):
        table.add_column(column, justify="right" if column != "candidate" else "left")
    for row in board["candidates"]:
        name = f"[bold]{row['id']}[/] ★" if row["id"] == champion else row["id"]
        table.add_row(
            str(row["rank"]), name, f"{num(row[f'cv_{metric}'])} ± {num(row['cv_std'])}",
            num(row[f"train_{metric}"]), str(row["complexity"]),
        )  # fmt: skip
    console.print(table)
    final, signoff, export = state.get("final"), state.get("signoff"), state.get("export")
    lines = []
    if final:
        consistency = final.get("cv_consistency") or {}
        lines.append(
            f"Champion [bold]{final['champion']}[/] ({final['family']}): test {metric} "
            f"[bold]{final['test'][metric]}[/] vs baseline {final['test']['baseline']}"
            + (
                " · within the CV range"
                if consistency.get("within_2_std_of_cv")
                else " · [yellow]outside the CV range[/]"
                if consistency
                else ""
            )
        )
    if signoff:
        lines.append(
            ("Trustworthy" if signoff["trustworthy"] else "[red]Not trustworthy[/]")
            + f": {escape(signoff['summary'])}"
        )
        lines += [f"  caveat: {escape(c)}" for c in signoff["caveats"]]
    for issue in state.get("unresolved_issues") or []:
        lines.append(f"[yellow]Unresolved ({issue['severity']}): {escape(issue['title'])}[/]")
    lines.append(f"Report: {run.root / 'report.md'}")
    if export:
        lines.append(f"Model bundle: {export['bundle_path']}")
        lines.append(f"  {export['usage']}")
    console.print(Panel("\n".join(lines), title=f"Result · {run.run_id}", border_style="green"))


def run_team(args: dict[str, Any], settings: Settings, console: Console) -> int:
    ws = Workspace(settings.workspace)
    options = TeamOptions(
        budget_minutes=args.get("budget_minutes") or 15.0,
        max_iterations=args.get("max_iterations") or 3,
        confirm_plan=not args.get("yes"),
    )
    graph, _ = build_team(settings, options, checkpointer=open_checkpointer(ws))
    thread_id = args.get("thread") or datetime.now().strftime("team-%Y%m%d-%H%M%S")
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    ui = TeamUI(console, show_thinking=not args.get("hide_thinking"))
    ui.approve_all = settings.auto_approve

    snapshot = graph.get_state(config)
    payload: Any
    if snapshot.values:
        if snapshot.interrupts:
            value = snapshot.interrupts[0].value
            console.print("[yellow]This run was waiting for your input.[/]")
            if value.get("type") == "confirm_plan":
                payload = Command(resume=ui.confirm_plan(value))
            else:
                payload = Command(resume={"decisions": ui.decide(value)})
        elif snapshot.next:
            console.print(f"[yellow]Resuming the run at: {', '.join(snapshot.next)}[/]")
            payload = None
        else:
            print_results(console, ws, snapshot.values)
            return 0
    elif not args.get("task"):
        console.print(
            '[red]Describe the task, e.g.[/] ml-agent team "Predict churned in data/customers.csv"'
        )
        return 2
    else:
        payload = {"request": args["task"]}

    console.print(
        Panel(
            f"[bold]ML agent team[/] · Planner → Data → Analysis → Modeling → Review → Visualization\n"
            f"model [cyan]{settings.model}[/] · budget {options.budget_minutes:g} min · "
            f"≤ {options.max_iterations} review rounds\n"
            f"workspace [cyan]{ws.root}[/] · thread [cyan]{thread_id}[/]"
            + (
                "\n[yellow]auto-approve is on: agent code runs without review[/]"
                if settings.auto_approve
                else ""
            ),
            border_style="green",
            expand=False,
        )
    )
    try:
        stream_team(graph, config, payload, ui)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Stopped. Resume with: ml-agent team --thread {thread_id}[/]")
        return 130
    state = graph.get_state(config).values
    if graph.get_state(config).next:
        console.print(f"[yellow]Run paused. Resume with: ml-agent team --thread {thread_id}[/]")
        return 0
    print_results(console, ws, state)
    console.print(f"[dim]Show these results again with: ml-agent team --thread {thread_id}[/]")
    return 0
