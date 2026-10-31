"""The agent team as a LangGraph ``StateGraph``.

    init_run → planner → confirm_plan (interrupt; feedback loops back to planner)
    → make_splits → data → analysis → modeling → review
    review: revise → data | analysis | modeling (at most ``max_iterations`` rounds)
            approve → final_evaluation → signoff → visualization → export

Specialists are LangChain ``create_agent`` agents with structured outputs. They share no chat
history; each receives a brief built from the earlier agents' reports. Each node invokes its
specialist exactly once (so approval interrupts inside a specialist resume cleanly); reports that
fail validation are sent back to the model inside that same invocation by ``ValidateReport``.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelCallLimitMiddleware,
    ToolErrorMiddleware,
    hook_config,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from ml_agent.agent import tool_error_message
from ml_agent.config import Settings, build_chat_model
from ml_agent.team import evaluation
from ml_agent.team.export import export_champion
from ml_agent.team.prompts import PROMPTS
from ml_agent.team.runs import RunStore
from ml_agent.team.schemas import (
    AnalysisReport,
    DataReport,
    ModelingReport,
    ProblemSpec,
    ReviewVerdict,
    SignOff,
    VisualReport,
)
from ml_agent.team.tools import (
    CODE_TOOLS,
    analysis_tools,
    data_tools,
    modeling_tools,
    planner_tools,
    review_tools,
    visualization_tools,
)
from ml_agent.workspace import Workspace

# Model calls each specialist may make per invocation.
CALL_LIMITS = {
    "planner": 15,
    "data": 25,
    "analysis": 20,
    "modeling": 30,
    "review": 20,
    "signoff": 6,
    "visualization": 25,
}
MAX_REPORT_REJECTIONS = 2
REJECTION_PREFIX = "Your report was rejected"


@dataclass(frozen=True)
class TeamOptions:
    budget_minutes: float = 15.0
    max_iterations: int = 3
    confirm_plan: bool = True


class TeamState(TypedDict, total=False):
    request: str
    run_id: str
    spec: dict
    plan_feedback: str | None
    splits: dict
    data_report: dict
    analysis_report: dict
    modeling_report: dict
    verdict: dict
    reviews: Annotated[list[dict], operator.add]
    iteration: int
    champion: str
    unresolved_issues: list[dict]
    final: dict
    signoff: dict
    visual: dict
    export: dict


class ValidateReport(AgentMiddleware):
    """Send an invalid structured report back to the model with the reason, a bounded number of times."""

    def __init__(self, validate: Callable[[Any], None]):
        super().__init__()
        self.validate = validate

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        report = state.get("structured_response")
        if report is None:
            return None
        try:
            self.validate(report)
        except (ValueError, KeyError, RuntimeError) as exc:
            rejections = sum(
                1
                for m in state["messages"]
                if isinstance(m, HumanMessage) and str(m.content).startswith(REJECTION_PREFIX)
            )
            if rejections >= MAX_REPORT_REJECTIONS:
                return None  # the node reports the failure
            detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
            return {
                "messages": [
                    HumanMessage(f"{REJECTION_PREFIX}: {detail}. Fix it and report again.")
                ],
                "structured_response": None,
                "jump_to": "model",
            }
        return None


def _format(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=1, default=str)


def brief(*sections: tuple[str, Any]) -> str:
    """The message a specialist receives: titled sections, skipping empty ones."""
    return "\n\n".join(f"## {title}\n{_format(value)}" for title, value in sections if value)


def build_team(
    settings: Settings,
    options: TeamOptions | None = None,
    *,
    models: BaseChatModel | dict[str, BaseChatModel] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile the team graph. ``models`` may be one model for all roles or one per role."""
    options = options or TeamOptions()
    ws = Workspace(settings.workspace)
    shared_model = None if isinstance(models, dict) else models
    caching = models is None and settings.provider == "anthropic"

    def model_for(role: str) -> BaseChatModel:
        nonlocal shared_model
        if isinstance(models, dict):
            return models[role]
        if shared_model is None:
            shared_model = build_chat_model(settings)
        return shared_model

    def ask(role: str, tools: list, schema: type[BaseModel], prompt: str, validate=None):
        """Invoke one specialist once and return its validated structured report."""
        middleware: list[AgentMiddleware] = [
            ModelCallLimitMiddleware(run_limit=CALL_LIMITS[role], exit_behavior="end")
        ]
        gated = [t.name for t in tools if t.name in CODE_TOOLS]
        if gated and not settings.auto_approve:
            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        name: InterruptOnConfig(
                            allowed_decisions=["approve", "edit", "reject"],
                            description=f"The {role} agent wants to run this code on your machine.",
                        )
                        for name in gated
                    }
                )
            )
        middleware.append(ToolErrorMiddleware(tool_error_message))
        if validate is not None:
            middleware.append(ValidateReport(validate))
        if caching:
            from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

            middleware.append(AnthropicPromptCachingMiddleware(ttl="1h"))
        agent = create_agent(
            model_for(role),
            tools,
            system_prompt=PROMPTS[role],
            middleware=middleware,
            response_format=schema,
            name=role,
        )
        result = agent.invoke({"messages": [HumanMessage(prompt)]})
        report = result.get("structured_response")
        if report is None:
            raise RuntimeError(f"The {role} agent stopped without its report (model-call limit?)")
        if validate is not None:
            validate(report)
        return report

    def open_run(state: TeamState) -> RunStore:
        return RunStore.open(ws, state["run_id"])

    def review_feedback(state: TeamState) -> dict | None:
        verdict = state.get("verdict")
        if verdict and verdict["decision"] == "revise":
            return {
                "revisit_stage": verdict["revisit_stage"],
                "requested_changes": verdict["requested_changes"],
                "issues": verdict["issues"],
            }
        return None

    # ------------------------------------------------------------------ nodes

    def init_run(state: TeamState) -> dict:
        if state.get("run_id"):
            return {}
        return {"run_id": RunStore.create(ws).run_id, "iteration": 1}

    def planner(state: TeamState) -> dict:
        run = open_run(state)
        spec = ask(
            "planner",
            planner_tools(ws, settings),
            ProblemSpec,
            brief(
                ("User request", state["request"]),
                ("Your previous plan", state.get("spec") if state.get("plan_feedback") else None),
                ("User feedback on that plan", state.get("plan_feedback")),
                ("Compute budget", f"{options.budget_minutes:g} minutes of model training"),
            ),
            validate=run.check_spec,
        )
        run.save_spec(spec, budget_seconds=options.budget_minutes * 60)
        return {"spec": spec.model_dump(), "plan_feedback": None}

    def confirm_plan(state: TeamState) -> Command:
        if not options.confirm_plan:
            return Command(goto="make_splits")
        answer = interrupt(
            {
                "type": "confirm_plan",
                "run_id": state["run_id"],
                "spec": state["spec"],
                "budget_minutes": options.budget_minutes,
                "max_iterations": options.max_iterations,
            }
        )
        feedback = (answer or {}).get("feedback")
        if feedback:
            return Command(goto="planner", update={"plan_feedback": feedback})
        return Command(goto="make_splits")

    def make_splits(state: TeamState) -> dict:
        run = open_run(state)
        summary = run.create_splits()
        run.update_meta(iteration=1)
        return {"splits": summary, "iteration": 1}

    def data(state: TeamState) -> dict:
        run = open_run(state)
        report = ask(
            "data",
            data_tools(run, settings),
            DataReport,
            brief(
                ("User request", state["request"]),
                ("Problem spec", state["spec"]),
                ("Splits", state["splits"]),
                ("Current column roles (inferred)", run.roles),
                ("Reviewer feedback to address", review_feedback(state)),
                (
                    "Your previous report",
                    state.get("data_report") if review_feedback(state) else None,
                ),
            ),
        )
        return {"data_report": report.model_dump()}

    def analysis(state: TeamState) -> dict:
        run = open_run(state)
        report = ask(
            "analysis",
            analysis_tools(run, settings),
            AnalysisReport,
            brief(
                ("Problem spec", state["spec"]),
                ("Data report", state["data_report"]),
                ("Column roles", run.roles),
                ("Reviewer feedback to address", review_feedback(state)),
            ),
        )
        return {"analysis_report": report.model_dump()}

    def modeling(state: TeamState) -> dict:
        run = open_run(state)

        def validate(report: ModelingReport) -> None:
            record = run.candidate(report.proposed_champion)
            if record["features_hash"] != run.features_hash:
                raise ValueError(f"{report.proposed_champion} was trained on an older feature set")

        report = ask(
            "modeling",
            modeling_tools(run, settings),
            ModelingReport,
            brief(
                ("Problem spec", state["spec"]),
                ("Data report", state["data_report"]),
                ("Analysis report", state["analysis_report"]),
                ("Round", f"{state['iteration']} of at most {options.max_iterations}"),
                ("Current leaderboard", evaluation.leaderboard(run)),
                ("Compute budget remaining", f"{run.remaining_seconds():.0f} seconds"),
                ("Reviewer feedback to address", review_feedback(state)),
            ),
            validate=validate,
        )
        return {"modeling_report": report.model_dump()}

    def review(state: TeamState) -> Command:
        run = open_run(state)
        iteration = state["iteration"]

        def validate(verdict: ReviewVerdict) -> None:
            if verdict.decision == "approve":
                if not verdict.champion:
                    raise ValueError("An approval must name the champion")
                record = run.candidate(verdict.champion)
                if record["features_hash"] != run.features_hash:
                    raise ValueError(f"{verdict.champion} was trained on an older feature set")
            elif verdict.revisit_stage is None:
                raise ValueError("A revision must name the stage to revisit")

        verdict = ask(
            "review",
            review_tools(run, settings),
            ReviewVerdict,
            brief(
                ("Problem spec", state["spec"]),
                ("Data report", state["data_report"]),
                ("Analysis report", state["analysis_report"]),
                ("Modeling report", state["modeling_report"]),
                ("Review round", f"{iteration} of at most {options.max_iterations}"),
                ("Earlier reviews", state.get("reviews")),
            ),
            validate=validate,
        )
        record = {"iteration": iteration, **verdict.model_dump()}
        run.write_json(f"reviews/iteration-{iteration}.json", record)
        update: dict[str, Any] = {"verdict": verdict.model_dump(), "reviews": [record]}
        if verdict.decision == "approve":
            return Command(goto="final_evaluation", update={**update, "champion": verdict.champion})
        if iteration >= options.max_iterations:
            # Out of rounds: evaluate the best available model and carry the open issues forward.
            champion = verdict.champion or evaluation.compare(run)["one_se_recommendation"]
            return Command(
                goto="final_evaluation",
                update={
                    **update,
                    "champion": champion,
                    "unresolved_issues": [issue.model_dump() for issue in verdict.issues],
                },
            )
        run.update_meta(iteration=iteration + 1)
        return Command(goto=verdict.revisit_stage, update={**update, "iteration": iteration + 1})

    def final_evaluation(state: TeamState) -> dict:
        return {"final": evaluation.final_evaluation(open_run(state), state["champion"])}

    def signoff(state: TeamState) -> dict:
        run = open_run(state)
        report = ask(
            "signoff",
            [],
            SignOff,
            brief(
                ("Problem spec", state["spec"]),
                ("Champion", run.candidate(state["champion"])),
                (
                    "Test-set evaluation",
                    {k: v for k, v in state["final"].items() if k != "permutation_importance"},
                ),
                ("Unresolved review issues", state.get("unresolved_issues")),
            ),
        )
        return {"signoff": report.model_dump()}

    def visualization(state: TeamState) -> dict:
        run = open_run(state)

        def validate(report: VisualReport) -> None:
            if not (run.root / "report.md").exists():
                raise ValueError("Write the report with write_report before finishing")

        report = ask(
            "visualization",
            visualization_tools(run, settings),
            VisualReport,
            brief(
                ("User request", state["request"]),
                ("Problem spec", state["spec"]),
                ("Data report", state["data_report"]),
                ("Analysis report", state["analysis_report"]),
                ("Modeling report", state["modeling_report"]),
                ("Review history", state.get("reviews")),
                ("Unresolved review issues", state.get("unresolved_issues")),
                ("Test-set evaluation", state["final"]),
                ("Reviewer sign-off", state["signoff"]),
            ),
            validate=validate,
        )
        return {"visual": report.model_dump()}

    def export(state: TeamState) -> dict:
        return {"export": export_champion(open_run(state))}

    graph = StateGraph(TeamState)
    graph.add_node("init_run", init_run)
    graph.add_node("planner", planner)
    graph.add_node("confirm_plan", confirm_plan, destinations=("planner", "make_splits"))
    graph.add_node("make_splits", make_splits)
    graph.add_node("data", data)
    graph.add_node("analysis", analysis)
    graph.add_node("modeling", modeling)
    graph.add_node(
        "review", review, destinations=("final_evaluation", "data", "analysis", "modeling")
    )
    graph.add_node("final_evaluation", final_evaluation)
    graph.add_node("signoff", signoff)
    graph.add_node("visualization", visualization)
    graph.add_node("export", export)
    for a, b in [
        (START, "init_run"),
        ("init_run", "planner"),
        ("planner", "confirm_plan"),
        ("make_splits", "data"),
        ("data", "analysis"),
        ("analysis", "modeling"),
        ("modeling", "review"),
        ("final_evaluation", "signoff"),
        ("signoff", "visualization"),
        ("visualization", "export"),
        ("export", END),
    ]:
        graph.add_edge(a, b)
    return graph.compile(checkpointer=checkpointer, name="ml_team"), ws
