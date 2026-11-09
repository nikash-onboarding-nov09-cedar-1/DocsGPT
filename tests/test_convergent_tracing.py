"""Unit coverage for opt-in chat tracing without exporters or external services."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from docsgpt.tracing import convergent as recording


@pytest.fixture
def telemetry(monkeypatch):
    """Replace only telemetry APIs, leaving the generator wrappers under test."""
    span = MagicMock()
    monkeypatch.setattr(recording, "initialize", lambda: True)
    monkeypatch.setattr(recording.convergent, "agent", lambda **kwargs: lambda func: func)
    monkeypatch.setattr(recording.convergent, "current_span", lambda: span)
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "span_only")
    return span


def agent():
    """Build the identity/configuration consumed by the chat wrapper."""
    return SimpleNamespace(upstream_model_id="fixture-model", prompt="Fixture prompt", tools=[],
                           tool_executor=SimpleNamespace(conversation_id=None))


def test_chat_deliverable_and_events_are_preserved(telemetry):
    events = [{"answer": "First "}, {"answer": "second"}, {"sources": [{"title": "Fixture"}]}]

    @recording.trace_chat
    def run(self, query):
        yield from events

    assert list(run(agent(), "Question")) == events
    telemetry.set_input.assert_called_once_with("Question")
    telemetry.set_output.assert_called_once_with({
        "answer": "First second", "sources": [{"title": "Fixture"}], "status": "completed"
    })
    assert not recording._chat_active.get()


def test_chat_close_records_partial_answer(telemetry):
    @recording.trace_chat
    def run(self, query):
        yield {"answer": "Partial"}
        yield {"answer": " not consumed"}

    stream = run(agent(), "Question")
    next(stream)
    stream.close()
    telemetry.set_output.assert_called_once_with({"answer": "Partial", "status": "interrupted"})
    assert not recording._chat_active.get()


def test_chat_failure_propagates_and_resets_scope(telemetry):
    @recording.trace_chat
    def run(self, query):
        yield {"answer": "Partial"}
        raise ValueError("fixture error")

    with pytest.raises(ValueError, match="fixture error"):
        list(run(agent(), "Question"))
    assert not recording._chat_active.get()


def test_content_opt_out(telemetry, monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")

    @recording.trace_chat
    def run(self, query):
        yield {"answer": "Private"}

    assert list(run(agent(), "Private")) == [{"answer": "Private"}]
    telemetry.set_input.assert_not_called()
    telemetry.set_output.assert_not_called()


def test_disabled_tracing_preserves_behavior(monkeypatch):
    monkeypatch.setattr(recording, "initialize", lambda: False)

    @recording.trace_chat
    def run(self, query):
        yield {"answer": query}

    assert list(run(agent(), "Answer")) == [{"answer": "Answer"}]


def test_tools_outside_chat_are_not_wrapped(monkeypatch):
    span_factory = MagicMock()
    monkeypatch.setattr(recording.convergent, "span", span_factory)

    @recording.trace_tool
    def run(self, tools_dict, call, llm_class_name):
        yield {"type": "pending"}
        return "Result", call.id

    stream = run(None, {}, SimpleNamespace(id="fixture-call"), "OpenAILLM")
    assert next(stream) == {"type": "pending"}
    with pytest.raises(StopIteration) as stopped:
        next(stream)
    assert stopped.value.value == ("Result", "fixture-call")
    span_factory.assert_not_called()


def test_tool_return_and_id_are_preserved(telemetry, monkeypatch):
    factory = MagicMock()
    factory.return_value.__enter__.return_value = telemetry
    monkeypatch.setattr(recording.convergent, "span", factory)
    token = recording._chat_active.set(True)

    @recording.trace_tool
    def run(self, tools_dict, call, llm_class_name):
        yield {"type": "pending"}
        return {"price": 42000}, call.id

    try:
        stream = run(None, {}, SimpleNamespace(id="fixture-call", name="price", arguments='{"symbol":"BTC"}'),
                     "OpenAILLM")
        assert next(stream) == {"type": "pending"}
        with pytest.raises(StopIteration) as stopped:
            next(stream)
        assert stopped.value.value == ({"price": 42000}, "fixture-call")
        telemetry.set_tool_call_id.assert_called_once_with("fixture-call")
        telemetry.set_output.assert_called_once_with({"price": 42000})
    finally:
        recording._chat_active.reset(token)


def test_initialize_requires_destination(monkeypatch):
    for key in ("CONVERGENT_API_KEY", "CONVERGENT_SPANS_DIR", "CONVERGENT_TRACES_EXPORTER"):
        monkeypatch.delenv(key, raising=False)
    init = MagicMock()
    monkeypatch.setattr(recording.convergent, "init", init)
    assert recording.initialize() is False
    init.assert_not_called()


@pytest.mark.parametrize("release", [None, "deployment-sha"])
def test_initialize_is_idempotent_and_preserves_settings(monkeypatch, release):
    monkeypatch.setattr(recording, "_initialized", False)
    monkeypatch.setenv("CONVERGENT_SPANS_DIR", "fixture-destination")
    monkeypatch.delenv("CONVERGENT_RELEASE", raising=False)
    if release:
        monkeypatch.setenv("CONVERGENT_RELEASE", release)
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    init = MagicMock(return_value=SimpleNamespace(enabled=True))
    instrumentor = MagicMock(is_instrumented_by_opentelemetry=False)
    monkeypatch.setattr(recording.convergent, "init", init)
    monkeypatch.setattr(recording, "OpenAIInstrumentor", lambda: instrumentor)
    assert recording.initialize() is True
    assert recording.initialize() is True
    init.assert_called_once_with(release=release or recording.get_version(), agents=["ClassicAgent"])
    instrumentor.instrument.assert_called_once()
    assert recording.capture_content() is False
