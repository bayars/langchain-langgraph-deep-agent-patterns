"""
Custom Langflow component — wraps the Aegra Deep Agent REST API.

Streams tokens live by calling self._event_manager.on_token() for each
chunk received from the Aegra SSE stream. Langflow sets _event_manager
on every component before execution when streaming is requested, so
each token is forwarded directly into the SSE pipeline.
"""

import json
import uuid

import requests as req

from langflow.custom import Component
from langflow.io import MessageTextInput, Output, StrInput
from langflow.schema.message import Message


class AegraDeepAgentComponent(Component):
    display_name = "Aegra Deep Agent"
    description = (
        "Run the Deep Agent via the Aegra REST API with live token streaming. "
        "Supports VFS file persistence, bash execution, and human-in-the-loop."
    )
    icon = "bot"
    name = "AegraDeepAgent"

    inputs = [
        StrInput(
            name="aegra_url",
            display_name="Aegra Server URL",
            value="http://template-server:8001",
            info="Base URL of the Aegra REST API (no trailing slash).",
            advanced=False,
        ),
        MessageTextInput(
            name="input_value",
            display_name="User Message",
            info="The message to send to the Deep Agent.",
        ),
    ]

    outputs = [
        Output(
            display_name="Response",
            name="output",
            method="run_agent",
        ),
    ]

    def _get_session_id(self) -> str | None:
        try:
            return self.graph.session_id or None
        except Exception:
            return None

    def _get_or_create_thread(self, url: str, session_id: str | None) -> str:
        cache_key = f"aegra_thread_{session_id}" if session_id else None
        if cache_key:
            existing = self._attributes.get(cache_key)
            if existing:
                try:
                    r = req.get(f"{url}/threads/{existing}", timeout=5)
                    if r.status_code == 200:
                        return existing
                except Exception:
                    pass

        meta = {"lf_session": session_id} if session_id else {}
        resp = req.post(f"{url}/threads", json={"metadata": meta}, timeout=10)
        resp.raise_for_status()
        thread_id: str = resp.json()["thread_id"]

        if cache_key:
            self._attributes[cache_key] = thread_id

        return thread_id

    def run_agent(self) -> Message:
        url = self.aegra_url.rstrip("/")
        user_message = str(self.input_value)
        session_id = self._get_session_id()
        thread_id = self._get_or_create_thread(url, session_id)

        message_id = str(uuid.uuid4())
        tokens: list[str] = []
        first_chunk = True

        stream_resp = req.post(
            f"{url}/threads/{thread_id}/runs/stream",
            json={
                "assistant_id": "deep",
                "input": {"messages": [{"role": "user", "content": user_message}]},
            },
            stream=True,
            timeout=180,
        )
        stream_resp.raise_for_status()

        pending_event = ""
        for raw_line in stream_resp.iter_lines(decode_unicode=True):
            if not raw_line:
                pending_event = ""
                continue
            if raw_line.startswith("event:"):
                pending_event = raw_line[6:].strip()
                continue
            if raw_line.startswith("data:"):
                data_str = raw_line[5:].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if pending_event == "messages/partial":
                    chunk = data.get("content", "")
                    if chunk:
                        tokens.append(chunk)
                        # Forward token into Langflow's SSE pipeline
                        if self._event_manager:
                            self._event_manager.on_token(
                                data={"chunk": chunk, "id": message_id}
                            )
                        first_chunk = False

                pending_event = ""

        return Message(text="".join(tokens))
