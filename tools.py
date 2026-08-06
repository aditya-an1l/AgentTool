"""
Tool definitions and execution logic for the agent.

The JSON schema list (`TOOL_DEFINITIONS`) is passed directly to the OpenAI
API so the model knows which functions are available. Each entry matches the
signature required by the specification.
"""

from __future__ import annotations

import difflib
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

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
            "name": "grep_search",
            "description": "Search files for a regex pattern, returning matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (defaults to current directory).",
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob pattern filtering which files to search in a directory.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff",
            "description": "Show a unified diff between two files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path_a": {
                        "type": "string",
                        "description": "First (original) file path.",
                    },
                    "path_b": {
                        "type": "string",
                        "description": "Second (changed) file path.",
                    },
                },
                "required": ["path_a", "path_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring in a file. Pass replace_all to change every occurrence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of failing on ambiguity.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
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
    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "Web search is unavailable: the 'ddgs' package is not installed. "
            "Install it with: pip install 'agenttool[search]'"
        )
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


def grep_search(pattern: str, path: str = ".", include: str = "*") -> str:
    """Return matching lines with `file:lineno:` prefixes for a regex pattern."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"Error: Invalid regex '{pattern}': {exc}"
    root = Path(path).expanduser()
    if not root.exists():
        return f"Error: Path not found - {root}"
    targets = [root] if root.is_file() else sorted(root.rglob(include))
    matches = []
    for target in targets:
        if not target.is_file():
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                matches.append(f"{target}:{lineno}: {line.rstrip()}")
    return "\n".join(matches) if matches else "No matches found."


def diff(path_a: str, path_b: str) -> str:
    """Return a unified diff between two files."""
    a = Path(path_a).expanduser()
    b = Path(path_b).expanduser()
    try:
        a_lines = a.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return f"Error reading {a}: {exc}"
    try:
        b_lines = b.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return f"Error reading {b}: {exc}"
    unified = list(difflib.unified_diff(a_lines, b_lines, fromfile=str(a), tofile=str(b)))
    return "\n".join(unified) if unified else "Files are identical."


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact substring in a file, erroring on ambiguous matches."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"Error: File not found - {p}"
    try:
        content = p.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading {p}: {exc}"
    if old_string not in content:
        return f"Error: old_string not found in {p}"
    if not replace_all and content.count(old_string) > 1:
        occurrences = content.count(old_string)
        msg = f"occurrences of old_string in {p}; pass replace_all=true or include more context."
        return f"Error: {occurrences} {msg}"
    try:
        p.write_text(content.replace(old_string, new_string), encoding="utf-8")
    except Exception as exc:
        return f"Error writing {p}: {exc}"
    return f"Successfully edited {p}"


_DANGEROUS_PATTERNS = [
    "rm -rf /", "rm -fr /", "rm -r /",
    ":(){ :|:& };:",
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
    if name == "grep_search":
        return grep_search(**args)
    if name == "diff":
        return diff(**args)
    if name == "edit_file":
        return edit_file(**args)
    if name == "run_command":
        return run_command(**args)
    return f"Unknown tool: {name}"
