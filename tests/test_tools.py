"""
Tests for tools.py — tool execution, file operations, command runner.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import (
    TOOL_DEFINITIONS,
    execute_tool,
    list_directory,
    read_file,
    run_command,
    write_file,
)


@pytest.fixture
def tmp_workspace() -> str:
    """Provide a temporary directory for file operations, cleaned up after."""
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestToolDefinitions:
    def test_has_expected_tools(self) -> None:
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        assert names == {
            "web_search",
            "read_file",
            "write_file",
            "list_directory",
            "run_command",
        }

    def test_every_tool_has_type_function(self) -> None:
        for t in TOOL_DEFINITIONS:
            assert t["type"] == "function"


class TestReadFile:
    def test_reads_existing_file(self, tmp_workspace: str) -> None:
        path = os.path.join(tmp_workspace, "hello.txt")
        with open(path, "w") as f:
            f.write("Hello, world!")
        result = read_file(path)
        assert result == "Hello, world!"

    def test_returns_error_for_missing_file(self) -> None:
        result = read_file("/tmp/nonexistent_file_12345")
        assert result.startswith("Error: File not found -")


class TestWriteFile:
    def test_writes_content_to_file(self, tmp_workspace: str) -> None:
        path = os.path.join(tmp_workspace, "output.txt")
        result = write_file(path, "new content")
        assert result == f"Successfully wrote to {path}"
        with open(path) as f:
            assert f.read() == "new content"

    def test_creates_parent_directories(self, tmp_workspace: str) -> None:
        path = os.path.join(tmp_workspace, "a", "b", "c", "deep.txt")
        result = write_file(path, "deep")
        assert "Successfully wrote" in result
        assert os.path.isfile(path)


class TestListDirectory:
    def test_lists_files_and_dirs(self, tmp_workspace: str) -> None:
        Path(tmp_workspace, "file_a.txt").write_text("a")
        Path(tmp_workspace, "file_b.txt").write_text("b")
        os.makedirs(os.path.join(tmp_workspace, "subdir"))
        result = list_directory(tmp_workspace)
        lines = result.split("\n")
        assert "file_a.txt" in lines
        assert "file_b.txt" in lines
        assert "subdir/" in lines

    def test_returns_error_for_non_directory(self) -> None:
        result = list_directory("/tmp/nonexistent_dir_abc123")
        assert result.startswith("Error: Not a directory -")


class TestRunCommand:
    def test_echo_stdout(self) -> None:
        result = run_command("echo hello world")
        assert result == "hello world"

    def test_captures_stderr(self) -> None:
        result = run_command("echo 'err msg' >&2; exit 1")
        assert "err msg" in result

    def test_timeout_on_hanging_command(self) -> None:
        result = run_command("sleep 60")  # well beyond 30s timeout
        assert "timed out" in result.lower()


class TestExecuteTool:
    def test_dispatches_web_search(self) -> None:
        with patch("tools.web_search", return_value="search results"):
            result = execute_tool("web_search", {"query": "test"})
            assert result == "search results"

    def test_dispatches_read_file(self) -> None:
        result = execute_tool("read_file", {"path": "/tmp/nonexistent"})
        assert result.startswith("Error: File not found -")

    def test_dispatches_write_file(self, tmp_workspace: str) -> None:
        path = os.path.join(tmp_workspace, "dispatch.txt")
        result = execute_tool("write_file", {"path": path, "content": "dispatch"})
        assert "Successfully wrote" in result

    def test_dispatches_list_directory(self) -> None:
        result = execute_tool("list_directory", {"path": os.path.dirname(__file__)})
        assert "test_tools.py" in result

    def test_dispatches_run_command(self) -> None:
        result = execute_tool("run_command", {"command": "echo dispatch"})
        assert result == "dispatch"

    def test_returns_error_for_unknown_tool(self) -> None:
        result = execute_tool("nonexistent_tool", {})
        assert result == "Unknown tool: nonexistent_tool"
