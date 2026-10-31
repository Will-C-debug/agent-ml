"""Agent-level tests: the real LangGraph agent and tools, driven by a scripted chat model."""

import json
from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.console import Console

from ml_agent.agent import build_agent
from ml_agent.cli import TerminalUI, repair_dangling_tool_calls
from ml_agent.config import Settings, build_chat_model


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed list of AI messages; tool binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages):
    return ScriptedModel(messages=iter(messages))


def call(name, args, call_id):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def tool_results(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


THREAD = {"configurable": {"thread_id": "test"}}


def test_full_workflow(settings, churn_csv):
    model = scripted(
        call("load_dataset", {"path": str(churn_csv), "name": "churn"}, "c1"),
        call("profile_dataset", {"name": "churn", "target": "churned"}, "c2"),
        call(
            "train_automl",
            {
                "dataset": "churn",
                "target": "churned",
                "time_budget": 5,
                "estimators": ["lgbm"],
                "drop_columns": ["customer_id", "signup_date", "cancel_reason", "constant"],
            },
            "c3",
        ),
        call("explain_model", {"experiment_id": "exp-001", "top_k": 3}, "c4"),
        call("list_experiments", {}, "c5"),
        AIMessage("exp-001 beats the baseline."),
    )
    agent, ws = build_agent(settings, model=model, checkpointer=InMemorySaver())
    result = agent.invoke({"messages": [HumanMessage("Predict churn")]}, THREAD)

    results = tool_results(result)
    assert [m.name for m in results] == [
        "load_dataset",
        "profile_dataset",
        "train_automl",
        "explain_model",
        "list_experiments",
    ]
    assert all(m.status != "error" for m in results), [m.content for m in results]
    assert "cancel_reason" in json.loads(results[1].content)["target"]["possible_leakage"]
    assert json.loads(results[2].content)["experiment_id"] == "exp-001"
    assert json.loads(results[4].content)[0]["experiment_id"] == "exp-001"
    assert result["messages"][-1].content == "exp-001 beats the baseline."
    assert ws.experiment_ids() == ["exp-001"]


def test_tool_errors_are_returned_to_the_model(settings):
    model = scripted(call("profile_dataset", {"name": "nope"}, "c1"), AIMessage("No such dataset."))
    agent, _ = build_agent(settings, model=model)
    result = agent.invoke({"messages": [HumanMessage("profile nope")]})
    (error,) = tool_results(result)
    assert error.status == "error"
    assert "Unknown dataset 'nope'" in error.content
    assert result["messages"][-1].content == "No such dataset."


def test_run_python_waits_for_approval(settings):
    model = scripted(call("run_python", {"code": "print(6 * 7)"}, "c1"), AIMessage("It is 42."))
    agent, _ = build_agent(settings, model=model, checkpointer=InMemorySaver())
    agent.invoke({"messages": [HumanMessage("compute")]}, THREAD)

    (pending,) = agent.get_state(THREAD).interrupts
    assert pending.value["action_requests"][0]["args"]["code"] == "print(6 * 7)"
    assert pending.value["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject"]

    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), THREAD)
    assert json.loads(tool_results(result)[-1].content)["stdout"].strip() == "42"
    assert result["messages"][-1].content == "It is 42."


def test_rejected_code_is_not_run(settings):
    model = scripted(call("run_python", {"code": "open('marker', 'w')"}, "c1"), AIMessage("OK."))
    agent, ws = build_agent(settings, model=model, checkpointer=InMemorySaver())
    agent.invoke({"messages": [HumanMessage("do it")]}, THREAD)
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "use profile_dataset"}]}),
        THREAD,
    )
    rejection = tool_results(result)[-1]
    assert rejection.status == "error"
    assert "use profile_dataset" in rejection.content
    assert not (ws.root / "marker").exists()


def test_edited_code_runs_instead(settings):
    model = scripted(call("run_python", {"code": "print('original')"}, "c1"), AIMessage("Done."))
    agent, _ = build_agent(settings, model=model, checkpointer=InMemorySaver())
    agent.invoke({"messages": [HumanMessage("run")]}, THREAD)
    edit = {
        "type": "edit",
        "edited_action": {"name": "run_python", "args": {"code": "print('edited')"}},
    }
    result = agent.invoke(Command(resume={"decisions": [edit]}), THREAD)
    # The middleware prepends a note telling the model its call was replaced.
    content = tool_results(result)[-1].content
    assert "human reviewer replaced this tool call" in content
    assert '"stdout": "edited\\n"' in content


def test_auto_approve_runs_without_interrupt(settings):
    model = scripted(call("run_python", {"code": "print('hi')"}, "c1"), AIMessage("Done."))
    agent, _ = build_agent(
        replace(settings, auto_approve=True), model=model, checkpointer=InMemorySaver()
    )
    result = agent.invoke({"messages": [HumanMessage("run")]}, THREAD)
    assert not agent.get_state(THREAD).interrupts
    assert json.loads(tool_results(result)[-1].content)["stdout"].strip() == "hi"


def test_repair_after_abandoned_approval(settings):
    model = scripted(
        call("run_python", {"code": "print(1)"}, "c1"),
        AIMessage("Starting over as asked."),
    )
    agent, _ = build_agent(settings, model=model, checkpointer=InMemorySaver())
    agent.invoke({"messages": [HumanMessage("run")]}, THREAD)
    assert agent.get_state(THREAD).interrupts

    repair_dangling_tool_calls(agent, THREAD)
    assert not agent.get_state(THREAD).interrupts
    result = agent.invoke({"messages": [HumanMessage("never mind")]}, THREAD)
    cancelled = tool_results(result)[-1]
    assert cancelled.tool_call_id == "c1" and cancelled.status == "error"
    assert result["messages"][-1].content == "Starting over as asked."


def test_terminal_ui_rendering():
    console = Console(record=True, width=120)
    ui = TerminalUI(console)
    ai = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Weighing the metric choice"},
            {"type": "text", "text": "Training a **baseline** now."},
        ],
        tool_calls=[
            {"name": "train_automl", "args": {"dataset": "churn", "target": "churned"}, "id": "t1"}
        ],
        id="ai-1",
    )
    record = {
        "experiment_id": "exp-001",
        "metric": "roc_auc",
        "best_estimator": "lgbm",
        "test": {"roc_auc": 0.81, "baseline": 0.5},
        "search": {"trials": 12, "seconds": 30.1},
    }
    ui.render_update({"model": {"messages": [ai]}})
    ui.render_update({"HumanInTheLoopMiddleware.after_model": {"messages": [ai]}})  # duplicate
    ui.render_update(
        {
            "tools": {
                "messages": [
                    ToolMessage(json.dumps(record), name="train_automl", tool_call_id="t1")
                ]
            }
        }
    )
    ui.render_update({"model": {"todos": [{"content": "Profile data", "status": "completed"}]}})
    text = console.export_text()
    assert "Weighing the metric choice" in text
    assert text.count("Training a baseline now.") == 1
    assert "train_automl" in text
    assert "exp-001 · best lgbm · held-out roc_auc 0.81 (baseline 0.5)" in text
    assert "Profile data" in text


def test_non_interactive_approval_rejects():
    ui = TerminalUI(Console(record=True))
    request = {
        "action_requests": [
            {"name": "run_python", "args": {"code": "print(1)"}, "description": "d"}
        ],
        "review_configs": [],
    }
    (decision,) = ui.decide(request)  # pytest's stdin is not a TTY
    assert decision["type"] == "reject"
    ui.approve_all = True
    assert ui.decide(request) == [{"type": "approve"}]


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_AGENT_MODEL", "ollama:qwen3:14b")
    monkeypatch.setenv("ML_AGENT_AUTO_APPROVE", "true")
    monkeypatch.setenv("ML_AGENT_MAX_TIME_BUDGET", "120")
    settings = Settings.from_env(workspace=str(tmp_path), effort=None)
    assert settings.model == "ollama:qwen3:14b"
    assert settings.provider == "ollama"
    assert settings.auto_approve is True
    assert settings.max_time_budget == 120
    assert settings.workspace == tmp_path
    with pytest.raises(ValueError):
        Settings.from_env(effort="extreme")


def test_claude_model_configuration(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = build_chat_model(Settings(effort="xhigh"))
    assert model.model == "claude-opus-5-5"
    assert model.reasoning_effort == "xhigh"
    assert model.streaming is True
    assert model.betas == ["server-side-fallback-2026-07-01"]
    assert model.model_kwargs == {"fallbacks": "default"}
    no_fallback = build_chat_model(Settings(anthropic_fallbacks=False))
    assert not no_fallback.betas and "fallbacks" not in no_fallback.model_kwargs


def test_ollama_model_configuration():
    pytest.importorskip("langchain_ollama")
    model = build_chat_model(Settings(model="ollama:qwen3:14b"))
    assert type(model).__name__ == "ChatOllama"
    assert model.model == "qwen3:14b"
    assert model.num_ctx == 32768
