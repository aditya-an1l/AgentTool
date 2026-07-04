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
            ("[Ollama] llama3.1:8b", "http://localhost:11434/v1"),
            ("[Ollama] qwen2.5-coder:7b", "http://localhost:11434/v1"),
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
            ("[LM Studio] qwen2.5-coder-7b-instruct", "http://localhost:1234/v1")
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
        mock_ollama.return_value = [("[Ollama] a", "http://localhost:11434/v1")]
        mock_lm.return_value = [("[LM Studio] b", "http://localhost:1234/v1")]
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
