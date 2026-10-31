"""The agent team graph, driven by scripted models (no LLM calls)."""

import json
from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ml_agent.team.graph import TeamOptions, build_team

FEATURES = (
    "import pandas as pd\n"
    "def prepare(df):\n"
    "    df = df.copy()\n"
    "    df['signup_month'] = pd.to_datetime(df['signup_date']).dt.month\n"
    "    return df\n"
)


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


_ids = iter(range(10_000))


def call(tool, /, **args):
    return AIMessage("", tool_calls=[{"name": tool, "args": args, "id": f"call-{next(_ids)}"}])


def spec(**overrides):
    fields = {
        "dataset": "churn",
        "target": "churned",
        "task": "classification",
        "primary_metric": "roc_auc",
        "secondary_metrics": ["accuracy"],
        "positive_class": "yes",
        "success_criteria": "Clearly beat the baseline",
        "min_primary_score": None,
        "cv_folds": 3,
        "test_size": 0.2,
        "group_column": None,
        "threshold_objective": None,
        "excluded_columns": ["customer_id", "cancel_reason"],
        "constraints": [],
        "assumptions": ["Rows are independent customers"],
    }
    return {**fields, **overrides}


def data_report():
    return call(
        "DataReport",
        summary="Parsed signup_date; excluded the identifier and the post-churn reason.",
        feature_engineering=["signup_month from signup_date"],
        column_roles_rationale=["plan is categorical"],
        leakage_findings=[{"title": "cancel_reason", "detail": "only filled after churn"}],
        data_quality_notes=["tenure is 10% missing"],
    )


def analysis_report():
    return call(
        "AnalysisReport",
        summary="tickets and tenure dominate.",
        key_drivers=[{"title": "tickets", "detail": "highest mutual information"}],
        patterns=[],
        hypotheses=["more tickets -> more churn"],
        recommended_families=["linear", "hist_gb"],
        preprocessing_advice=[],
    )


def modeling_report(champion, tried):
    return call(
        "ModelingReport",
        summary="Searched several families.",
        candidates_tried=tried,
        proposed_champion=champion,
        rationale="Best mean CV score.",
        open_questions=[],
    )


def verdict(decision, champion=None, stage=None, changes=()):
    return call(
        "ReviewVerdict",
        decision=decision,
        revisit_stage=stage,
        champion=champion,
        issues=[]
        if decision == "approve"
        else [{"severity": "major", "title": "Narrow search", "evidence": "only 3 families"}],
        requested_changes=list(changes),
        summary=f"{decision}",
    )


SIGNOFF = call(
    "SignOff", trustworthy=True, summary="Test score matches CV.", caveats=["small test set"]
)


def visual_report():
    return [
        call("make_chart", kind="leaderboard"),
        call("write_report", markdown="# Churn model\n\n![Leaderboard](figures/leaderboard.png)\n"),
        call(
            "VisualReport",
            headline="A model that beats the baseline",
            insights=["tickets drive churn"],
            recommended_actions=["contact customers with many tickets"],
            figures=[{"path": "figures/leaderboard.png", "caption": "Leaderboard"}],
            report_path="report.md",
        ),
    ]


def models(**scripts):
    return {role: Scripted(messages=iter(messages)) for role, messages in scripts.items()}


def drive(graph, config, payload, answer):
    """Run until the graph finishes, answering every interrupt with ``answer(value)``."""
    graph.invoke(payload, config)
    seen = []
    while pending := graph.get_state(config).interrupts:
        value = pending[0].value
        seen.append(value)
        graph.invoke(Command(resume=answer(value)), config)
    return graph.get_state(config).values, seen


@pytest.fixture
def team_settings(settings, churn_csv, ws):
    ws.import_file(str(churn_csv), "churn")
    return settings


def test_full_team_run(team_settings):
    scripts = models(
        planner=[
            call("preview_dataset", name="churn", n_rows=3),
            call("ProblemSpec", **spec()),
            call("ProblemSpec", **spec(secondary_metrics=["accuracy", "ap"])),
        ],
        data=[call("define_features", code=FEATURES), call("get_column_roles"), data_report()],
        analysis=[
            call("feature_associations", top_k=5),
            call("quick_baselines"),
            analysis_report(),
        ],
        modeling=[
            call(
                "tune_families",
                families=["dummy", "linear", "hist_gb"],
                seconds_per_family=3,
                max_configs=2,
            ),
            modeling_report("c03-hist_gb", ["c01-dummy", "c02-linear", "c03-hist_gb"]),
            call("tune_families", families=["random_forest"], seconds_per_family=3, max_configs=2),
            # First proposal names a candidate that doesn't exist; the validator sends it back.
            modeling_report("c99-imaginary", ["c04-random_forest"]),
            modeling_report("c04-random_forest", ["c04-random_forest"]),
        ],
        review=[
            call("compare_candidates"),
            verdict("revise", stage="modeling", changes=["Also try random_forest"]),
            call("check_leakage"),
            verdict("approve", champion="c02-linear"),
        ],
        signoff=[SIGNOFF],
        visualization=visual_report(),
    )
    graph, ws = build_team(
        team_settings, TeamOptions(budget_minutes=5), models=scripts, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "t1"}}

    def answer(value):
        if value.get("type") == "confirm_plan":
            if "ap" not in value["spec"]["secondary_metrics"]:
                return {"feedback": "Also report average precision"}
            return {"approve": True}
        assert value["action_requests"][0]["name"] == "define_features"
        return {"decisions": [{"type": "approve"}]}

    state, interrupts = drive(graph, config, {"request": "Predict churn in churn.csv"}, answer)
    kinds = [v.get("type", v.get("action_requests", [{}])[0].get("name")) for v in interrupts]
    assert kinds == ["confirm_plan", "confirm_plan", "define_features"]

    assert state["spec"]["secondary_metrics"] == ["accuracy", "ap"]
    assert [r["decision"] for r in state["reviews"]] == ["revise", "approve"]
    assert state["iteration"] == 2
    assert state["modeling_report"]["proposed_champion"] == "c04-random_forest"
    assert state["champion"] == "c02-linear"  # the reviewer may pick a simpler model
    assert state["final"]["champion"] == "c02-linear" and state["final"]["beats_baseline"] is True
    assert state["signoff"]["trustworthy"] is True

    run_dir = ws.runs_dir / state["run_id"]
    report = (run_dir / "report.md").read_text()
    assert "Appendix: leaderboard" in report and "c04-random_forest" in report
    assert (run_dir / "figures" / "leaderboard.png").exists()
    assert (run_dir / "reviews" / "iteration-1.json").exists()
    assert "signup_month" in json.loads((run_dir / "data.json").read_text())["roles"]["numeric"]
    assert set(state["export"]["files"]) >= {"model.joblib", "predict.py", "features.py"}


def test_iteration_cap_and_rejected_code(team_settings):
    scripts = models(
        planner=[call("ProblemSpec", **spec())],
        data=[call("run_python", code="import os; os.remove('important.txt')"), data_report()],
        analysis=[analysis_report()],
        modeling=[
            call(
                "tune_families", families=["dummy", "linear"], seconds_per_family=2, max_configs=2
            ),
            modeling_report("c02-linear", ["c01-dummy", "c02-linear"]),
        ],
        review=[verdict("revise", stage="modeling", changes=["search more"])],
        signoff=[SIGNOFF],
        visualization=visual_report(),
    )
    options = TeamOptions(budget_minutes=5, max_iterations=1, confirm_plan=False)
    graph, ws = build_team(team_settings, options, models=scripts, checkpointer=InMemorySaver())
    (ws.root / "important.txt").write_text("keep me")
    config = {"configurable": {"thread_id": "t2"}}

    def answer(value):
        assert value["action_requests"][0]["name"] == "run_python"
        return {"decisions": [{"type": "reject", "message": "Don't delete files"}]}

    state, interrupts = drive(graph, config, {"request": "Predict churn"}, answer)
    assert len(interrupts) == 1
    assert (ws.root / "important.txt").exists()
    # Out of review rounds: the best available model is evaluated, with the issues carried along.
    assert state["champion"] == "c02-linear"
    assert state["unresolved_issues"][0]["title"] == "Narrow search"
    assert state["final"]["champion"] == "c02-linear"


def test_resume_after_a_crash(team_settings):
    class Exploding(Scripted):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("network down")

    base = dict(
        planner=[call("ProblemSpec", **spec())],
        data=[data_report()],
        modeling=[
            call(
                "tune_families", families=["dummy", "linear"], seconds_per_family=2, max_configs=2
            ),
            modeling_report("c02-linear", ["c01-dummy", "c02-linear"]),
        ],
        review=[verdict("approve", champion="c02-linear")],
        signoff=[SIGNOFF],
        visualization=visual_report(),
    )
    options = TeamOptions(budget_minutes=5, confirm_plan=False)
    settings = replace(team_settings, auto_approve=True)
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t3"}}
    crashing = {**models(**base), "analysis": Exploding(messages=iter([]))}
    graph, _ = build_team(settings, options, models=crashing, checkpointer=saver)
    with pytest.raises(ConnectionError):
        graph.invoke({"request": "Predict churn"}, config)
    assert graph.get_state(config).next == ("analysis",)

    # A fresh process resumes from the checkpoint: planner and data agents are not re-run.
    resumed = {
        **models(**{k: [] for k in ("planner", "data")}),
        **{k: v for k, v in models(**base).items() if k not in ("planner", "data")},
        "analysis": Scripted(messages=iter([analysis_report()])),
    }
    graph, _ = build_team(settings, options, models=resumed, checkpointer=saver)
    graph.invoke(None, config)
    state = graph.get_state(config).values
    assert state["final"]["champion"] == "c02-linear" and "export" in state


def test_tool_messages_reach_agents(team_settings):
    """Tool errors come back to the model instead of crashing the run."""
    scripts = models(
        planner=[
            call("profile_dataset", name="does-not-exist"),
            call("ProblemSpec", **spec()),
        ],
    )
    graph, _ = build_team(
        team_settings, TeamOptions(confirm_plan=True), models=scripts, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "t4"}}
    graph.invoke({"request": "Predict churn"}, config)
    (pending,) = graph.get_state(config).interrupts
    assert pending.value["type"] == "confirm_plan"
    assert pending.value["spec"]["target"] == "churned"


def test_invalid_plan_is_sent_back_to_the_planner(team_settings):
    scripts = models(
        planner=[
            call("ProblemSpec", **spec(target="no_such_column")),
            call("ProblemSpec", **spec()),
        ],
    )
    graph, _ = build_team(team_settings, models=scripts, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t5"}}
    graph.invoke({"request": "Predict churn"}, config)
    assert graph.get_state(config).values["spec"]["target"] == "churned"
    # Both scripted replies were consumed: the first plan was rejected and the planner retried.
    assert next(scripts["planner"].messages, None) is None


def test_team_ui_rendering():
    from rich.console import Console

    from ml_agent.team.cli import TeamUI

    console = Console(record=True, width=140)
    ui = TeamUI(console)
    report_call = call("DataReport", summary="hidden")
    ui.agent_update(("data:abc",), {"model": {"messages": [report_call]}})
    ui.agent_update(
        ("data:abc",),
        {"model": {"messages": [AIMessage('{"summary": "json report"}', id="json-1")]}},
    )
    ui.stage_update("data", {"data_report": {"summary": "Parsed dates."}})
    ui.stage_update(
        "review",
        {"verdict": {"decision": "revise", "revisit_stage": "modeling", "summary": "Too narrow",
                     "requested_changes": ["try hist_gb"]}},
    )  # fmt: skip
    plan = {"type": "confirm_plan", "spec": spec(), "budget_minutes": 10, "max_iterations": 3}
    assert ui.confirm_plan(plan) == {"approve": True}  # stdin is not a TTY under pytest
    text = console.export_text()
    assert "Data agent · access & preparation" in text
    assert "DataReport" not in text and "json report" not in text
    assert "Parsed dates." in text and "try hist_gb" in text
    assert "Proposed plan" in text and "customer_id, cancel_reason" in text
