"""
Tests for agent.py — model discovery, tool parsing, client creation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import (
    _call_model,
    _fetch_lmstudio_models,
    _fetch_ollama_models,
    discover_models,
    format_tool_result,
    robust_tool_parse,
)


class TestFetchOllamaModels:
    @patch("agent.requests.get")
    def test_returns_formatted_models(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5-coder:7b"}]
        }
        models = _fetch_ollama_models()
        assert models == [
            ("[Ollama] llama3.1:8b", "llama3.1:8b", "http://localhost:11434/v1"),
            ("[Ollama] qwen2.5-coder:7b", "qwen2.5-coder:7b", "http://localhost:11434/v1"),
        ]

    @patch("agent.requests.get", side_effect=Exception("Connection refused"))
    def test_returns_empty_when_offline(self, mock_get: MagicMock) -> None:
        assert _fetch_ollama_models() == []


class TestFetchLmStudioModels:
    @patch("agent.requests.get")
    def test_returns_formatted_models(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "data": [{"id": "qwen2.5-coder-7b-instruct"}]
        }
        models = _fetch_lmstudio_models()
        assert models == [
            ("[LM Studio] qwen2.5-coder-7b-instruct", "qwen2.5-coder-7b-instruct", "http://localhost:1234/v1")
        ]

    @patch("agent.requests.get", side_effect=Exception("Connection refused"))
    def test_returns_empty_when_offline(self, mock_get: MagicMock) -> None:
        assert _fetch_lmstudio_models() == []


class TestDiscoverModels:
    @patch("agent._fetch_ollama_models")
    @patch("agent._fetch_lmstudio_models")
    def test_combines_both_servers(
        self,
        mock_lm: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        mock_ollama.return_value = [("[Ollama] a", "a", "http://localhost:11434/v1")]
        mock_lm.return_value = [("[LM Studio] b", "b", "http://localhost:1234/v1")]
        result = discover_models()
        assert len(result) == 2

    @patch("agent._fetch_ollama_models", return_value=[])
    @patch("agent._fetch_lmstudio_models", return_value=[])
    def test_returns_empty_when_none_running(
        self,
        mock_lm: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        assert discover_models() == []


class TestRobustToolParse:
    def test_returns_official_tool_calls(self) -> None:
        msg: Dict[str, Any] = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x.py"}'},
                }
            ]
        }
        result = robust_tool_parse(msg)
        assert result == msg["tool_calls"]

    def test_fallback_json_in_content(self) -> None:
        msg: Dict[str, Any] = {
            "content": 'Some text {"name": "web_search", "arguments": {"query": "hello"}} more text'
        }
        result = robust_tool_parse(msg)
        assert len(result) == 1
        assert result[0]["id"] == "fallback"
        assert result[0]["function"]["name"] == "web_search"

    def test_no_json_returns_empty_list(self) -> None:
        msg: Dict[str, Any] = {"content": "Just a plain text reply."}
        assert robust_tool_parse(msg) == []

    def test_empty_message_returns_empty(self) -> None:
        assert robust_tool_parse({}) == []

    def test_missing_arguments_in_json_fallback(self) -> None:
        msg: Dict[str, Any] = {
            "content": '{"name": "web_search"}'  # missing "arguments"
        }
        assert robust_tool_parse(msg) == []

    def test_fallback_func_name_parens(self) -> None:
        msg: Dict[str, Any] = {
            "content": 'web_search({"query": "current president of India"})'
        }
        result = robust_tool_parse(msg)
        assert len(result) == 1
        assert result[0]["id"] == "fallback"
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "current president of India"}

    def test_fallback_tool_invocation(self) -> None:
        msg: Dict[str, Any] = {
            "content": '{"toolInvocation": "web_search", "functionCall": {"name": "web_search", "parameters": {"query": "owner of this model"}}}'
        }
        result = robust_tool_parse(msg)
        assert len(result) == 1
        assert result[0]["id"] == "fallback"
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "owner of this model"}

    def test_fallback_tool_invocation_list_wrapped(self) -> None:
        msg: Dict[str, Any] = {
            "content": '[{"toolInvocation": "web_search", "functionCall": {"name": "web_search", "parameters": {"query": "test"}}}]'
        }
        result = robust_tool_parse(msg)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "test"}

    def test_fallback_code_style_colon(self) -> None:
        msg: Dict[str, Any] = {
            "content": 'web_search(query: "who is the president of France")'
        }
        result = robust_tool_parse(msg)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "who is the president of France"}

    def test_fallback_code_style_equals(self) -> None:
        msg: Dict[str, Any] = {
            "content": 'read_file(path="test.txt")'
        }
        result = robust_tool_parse(msg)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "read_file"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"path": "test.txt"}


class TestFormatToolResult:
    def test_returns_json_string(self) -> None:
        result = format_tool_result("read_file", "file content")
        parsed = json.loads(result)
        assert parsed == {"tool_name": "read_file", "result": "file content"}

    def test_handles_dict_result(self) -> None:
        result = format_tool_result("web_search", {"count": 5})
        parsed = json.loads(result)
        assert parsed["tool_name"] == "web_search"
        assert parsed["result"]["count"] == 5


# NOTE: I noticed that the create_openai_client is a trivial 4-line factory
# that imports openai lazily and sets two attributes.  It does not warrant
# a unit test its behavior is exercised through the agent-loop integration path.


class MockDelta:
    """Simulates openai.types.chat.chat_completion_chunk.ChoiceDelta."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[MagicMock] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Simulates openai.types.chat.chat_completion_chunk.Choice."""

    def __init__(
        self, delta: MockDelta, finish_reason: str | None = None
    ) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class MockChunk:
    """Simulates one delta from an OpenAI streaming response."""

    def __init__(self, choices: list[MockChoice]) -> None:
        self.choices = choices


def _tool_call_delta(
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> MagicMock:
    """Build a mock tool-call delta for a streaming chunk."""
    tc = MagicMock()
    tc.index = index
    tc.id = id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class TestCallModelStreaming:
    """Test _call_model with streaming responses."""

    @patch("agent.Live")
    @patch("agent._process_tool_calls")
    def test_streams_text_content(
        self,
        mock_process: MagicMock,
        mock_live: MagicMock,
    ) -> None:
        chunks = [
            MockChunk([MockChoice(MockDelta("Hello "))]),
            MockChunk([MockChoice(MockDelta("world!"))]),
            MockChunk(
                [MockChoice(MockDelta(None), finish_reason="stop")]
            ),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        result = _call_model(
            mock_client, "test-model", [{"role": "user", "content": "hi"}]
        )

        assert result == "Hello world!"

    @patch("agent.Live")
    @patch("agent._process_tool_calls")
    def test_streams_tool_calls(
        self,
        mock_process: MagicMock,
        mock_live: MagicMock,
    ) -> None:
        chunks = [
            MockChunk(
                [
                    MockChoice(
                        MockDelta(
                            tool_calls=[
                                _tool_call_delta(
                                    index=0,
                                    id="call_1",
                                    name="web_search",
                                    arguments='{"query": ',
                                )
                            ]
                        )
                    )
                ]
            ),
            MockChunk(
                [
                    MockChoice(
                        MockDelta(
                            tool_calls=[
                                _tool_call_delta(
                                    index=0,
                                    arguments='"test"}',
                                )
                            ]
                        )
                    )
                ]
            ),
            MockChunk(
                [MockChoice(MockDelta(None), finish_reason="tool_calls")]
            ),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        result = _call_model(
            mock_client, "test-model", [{"role": "user", "content": "search"}]
        )

        assert result is None
        # Verify the tool call was reassembled and dispatched
        assert mock_process.called
        call_args = mock_process.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["function"]["name"] == "web_search"
        assert call_args[0]["function"]["arguments"] == '{"query": "test"}'

    @patch("agent.Live")
    @patch("agent._process_tool_calls")
    def test_streams_mixed_content_and_tool_calls(
        self,
        mock_process: MagicMock,
        mock_live: MagicMock,
    ) -> None:
        chunks = [
            MockChunk([MockChoice(MockDelta("Thinking..."))]),
            MockChunk(
                [
                    MockChoice(
                        MockDelta(
                            tool_calls=[
                                _tool_call_delta(
                                    index=0,
                                    id="call_1",
                                    name="read_file",
                                    arguments='{"path": "',
                                )
                            ]
                        )
                    )
                ]
            ),
            MockChunk(
                [
                    MockChoice(
                        MockDelta(
                            tool_calls=[
                                _tool_call_delta(
                                    index=0, arguments='test.txt"}'
                                )
                            ]
                        )
                    )
                ]
            ),
            MockChunk(
                [MockChoice(MockDelta(None), finish_reason="tool_calls")]
            ),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        result = _call_model(
            mock_client,
            "test-model",
            [{"role": "user", "content": "read file"}],
        )

        assert result is None
        assert mock_process.called
        call_args = mock_process.call_args[0][0]
        assert call_args[0]["function"]["name"] == "read_file"

    @patch("agent.Live")
    def test_streams_empty_response(self, mock_live: MagicMock) -> None:
        chunks = [
            MockChunk([MockChoice(MockDelta(None), finish_reason="stop")])
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        result = _call_model(
            mock_client,
            "test-model",
            [{"role": "user", "content": "empty"}],
        )

        assert result is None


# NOTE: I noticed that the create_openai_client is a trivial 4-line factory
# that imports openai lazily and sets two attributes.  It does not warrant
# a unit test its behavior is exercised through the agent-loop integration path.
