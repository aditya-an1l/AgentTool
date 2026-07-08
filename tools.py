"""
Tool definitions and execution logic for the agent.

The JSON schema list (`TOOL_DEFINITIONS`) is passed directly to the OpenAI
API so the model knows which functions are available. Each entry matches the
signature required by the specification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ddgs import DDGS

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo and return the top 5 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the supplied content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path (directories are created automatically).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and sub-folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and capture stdout + stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The full command to run (no interactive prompts).",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def web_search(query: str) -> str:
    """Return top 5 DuckDuckGo results formatted as a human-readable string."""
    results = []
    with DDGS() as ddg:
        for r in ddg.text(
            query, region="wt-wt", safesearch="Moderate", timelimit="w", max_results=5
        ):
            results.append(f"- **{r['title']}**\n  {r['href']}\n  {r['body']}\n")
    if not results:
        return "No results found."
    return "\n".join(results)


def read_file(path: str) -> str:
    """Read a file, returning its content or an error message."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"Error: File not found - {p}"
    try:
        return p.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading {p}: {exc}"


def write_file(path: str, content: str) -> str:
    """Write *content* to *path*, creating parent directories as needed."""
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {p}"
    except Exception as exc:
        return f"Error writing {p}: {exc}"


def list_directory(path: str) -> str:
    """Return a newline-separated listing of files/folders."""
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"Error: Not a directory - {p}"
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        lines = []
        for e in entries:
            suffix = "/" if e.is_dir() else ""
            lines.append(f"{e.name}{suffix}")
        return "\n".join(lines) if lines else "Directory is empty."
    except Exception as exc:
        return f"Error listing {p}: {exc}"


_DANGEROUS_PATTERNS = [
    "rm -rf", "rm -fr", "rm -r /",
    ">", ">>", "dd if=", ":(){ :|:& };:",
    "mkfs.", "fdisk", "dd if=",
]


def run_command(command: str) -> str:
    """Execute a command with a 30-second timeout."""
    cmd_lower = command.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.lower() in cmd_lower:
            return f"Error: Command blocked — contains dangerous pattern '{pattern}'."
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = completed.stdout.strip()
        err = completed.stderr.strip()
        if completed.returncode != 0:
            return f"[stderr]\n{err}\n[stdout]\n{out}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as exc:
        return f"Error executing command: {exc}"


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Map a tool name to its implementation and return the result."""
    if name == "web_search":
        return web_search(**args)
    if name == "read_file":
        return read_file(**args)
    if name == "write_file":
        return write_file(**args)
    if name == "list_directory":
        return list_directory(**args)
    if name == "run_command":
        return run_command(**args)
    return f"Unknown tool: {name}"
