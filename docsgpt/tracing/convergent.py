"""Opt-in Convergent recording for the streamed classic chat agent.

Enabled by CONVERGENT_API_KEY or a local/console destination. Content is
captured by default when enabled; the standard OTel content switch can opt out.
Existing DocsGPT traces and OpenTelemetry exporters are left in place.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Generator

import convergent
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

from docsgpt.version import get_version

_lock = threading.Lock()
_initialized = False
_chat_active: ContextVar[bool] = ContextVar("convergent_chat_active", default=False)


def initialize() -> bool:
    """Initialize once per process, only when a Convergent destination is set."""
    global _initialized
    if not any(os.getenv(key) for key in (
        "CONVERGENT_API_KEY", "CONVERGENT_SPANS_DIR", "CONVERGENT_TRACES_EXPORTER"
    )):
        return False
    with _lock:
        if not _initialized:
            # Use the application's release source, including in wheels/images.
            status = convergent.init(
                release=os.getenv("CONVERGENT_RELEASE") or get_version(), agents=["ClassicAgent"]
            )
            if not status.enabled:
                return False
            # Respect explicit OTel settings. These defaults put model content on
            # spans rather than log events, which Convergent does not ingest.
            os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
            os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "span_only")
            instrumentor = OpenAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            _initialized = True
    return True


def capture_content() -> bool:
    """Honor the standard OTel content opt-out for hand-written spans too."""
    return os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "").lower() in {
        "true", "span_only", "span_and_event"
    }


def trace_chat(func: Callable) -> Callable:
    """Trace the whole generator, retaining the answer actually sent to its reader."""
    @convergent.agent(name="ClassicAgent")
    def recorded(self: Any, query: str, *args: Any, **kwargs: Any) -> Generator:
        span = convergent.current_span()
        capture = capture_content()
        span.set_attribute("gen_ai.request.model", self.upstream_model_id)
        if capture:
            span.set_input(query)
            span.set_attribute("gen_ai.system_instructions", self.prompt)
        answer = []
        deliverable: dict[str, Any] = {"status": "interrupted"}
        try:
            for event in func(self, query, *args, **kwargs):
                if capture and isinstance(event, dict):
                    if "answer" in event:
                        answer.append(event["answer"])
                    for key in ("sources", "tool_calls", "metadata", "error"):
                        if key in event:
                            deliverable[key] = event[key]
                yield event
            deliverable["status"] = "error" if "error" in deliverable else "completed"
            if any(call.get("status") in {"awaiting_approval", "requires_client_execution"}
                   for call in deliverable.get("tool_calls", [])):
                deliverable["status"] = "paused"
        finally:
            if capture:
                deliverable["answer"] = "".join(answer)
                span.set_output(deliverable)
                if self.tools:
                    span.set_attribute("gen_ai.tool.definitions", json.dumps(self.tools))

    @wraps(func)
    def wrapper(self: Any, query: str, *args: Any, **kwargs: Any) -> Generator:
        if not initialize():
            yield from func(self, query, *args, **kwargs)
            return
        conversation_id = getattr(self.tool_executor, "conversation_id", None)
        scope = convergent.session(str(conversation_id)) if conversation_id else nullcontext()
        with scope:
            token = _chat_active.set(True)
            try:
                yield from recorded(self, query, *args, **kwargs)
            finally:
                _chat_active.reset(token)

    return wrapper


def trace_tool(func: Callable) -> Callable:
    """Record one complete tool generator, including its return value and call ID."""
    @wraps(func)
    def wrapper(self: Any, tools_dict: dict, call: Any, llm_class_name: str) -> Generator:
        # ToolExecutor is shared with other agents. Only the requested chat
        # agent's work should acquire a new tool span.
        if not _chat_active.get():
            return (yield from func(self, tools_dict, call, llm_class_name))
        with convergent.span(name=call.name, operation="tool_call") as span:
            span.set_tool_call_id(call.id)
            capture = capture_content()
            if capture:
                span.set_input(call.arguments)
            outcome = yield from func(self, tools_dict, call, llm_class_name)
            if capture and outcome is not None:
                span.set_output(outcome[0])
            return outcome
    return wrapper
