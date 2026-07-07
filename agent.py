#!/usr/bin/env python3
"""
AgentTool: Terminal-based Agentic Coding Assistant.

* Discovers local LLM servers (Ollama & LM Studio).
* Lets the user pick a model via a Rich UI.
* Runs an OpenAI-compatible chat loop with tool-calling.
* Supports web search, file read/write, directory listing and command execution.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Tuple, Optional

import requests
from rich import box
from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import FormattedText
from rich.traceback import install as install_rich_traceback

from tools import (
    TOOL_DEFINITIONS,
    execute_tool,
)

install_rich_traceback(show_locals=False)

console = Console()


def _fetch_ollama_models() -> List[Tuple[str, str, str]]:
    """
    Return a list of (display_name, model_id, base_url) for Ollama.
    """
    url = "http://localhost:11434/api/tags"
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
    except Exception:
        return []  # Ollama not running / unreachable

    data = resp.json()
    models = [
        (f"[Ollama] {m['name']}", m["name"], "http://localhost:11434/v1")
        for m in data.get("models", [])
    ]
    return models


def _fetch_lmstudio_models() -> List[Tuple[str, str, str]]:
    """
    Return a list of (display_name, model_id, base_url) for LM Studio.
    """
    url = "http://localhost:1234/v1/models"
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
    except Exception:
        return []  # NOTE: LM Studio not running / unreachable

    data = resp.json()
    # NOTE: LM Studio returns {"data": [{"id": "...", ...}], ...}
    models = [
        (f"[LM Studio] {m['id']}", m["id"], "http://localhost:1234/v1")
        for m in data.get("data", [])
    ]
    return models


def discover_models() -> List[Tuple[str, str, str]]:
    """Combine Ollama and LM Studio model lists."""
    ollama = _fetch_ollama_models()
    lmstudio = _fetch_lmstudio_models()
    return ollama + lmstudio


def _display_models(models: List[Tuple[str, str, str]]) -> None:
    table = Table(title="Discovered Local LLM Models", box=box.SIMPLE)
    table.add_column("#", justify="right")
    table.add_column("Model", overflow="fold")
    for idx, (name, _, _) in enumerate(models, start=1):
        table.add_row(str(idx), name)
    console.print(table)


def pick_model(models: List[Tuple[str, str, str]]) -> Tuple[str, str, str]:
    """Prompt the user to select a model; returns (display_name, model_id, base_url)."""
    while True:
        try:
            choice = Prompt.ask(
                "\nSelect a model by number (or 'q' to quit)",
                default="1",
            )
            if choice.lower() in {"q", "quit", "exit"}:
                console.print("[bold red]User aborted model selection.[/]")
                sys.exit(0)

            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                console.print("[red]Invalid number - try again.[/]")
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted - exiting.[/]")
            sys.exit(0)
        except Exception:
            console.print("[red]Please enter a valid number.[/]")


def create_openai_client(base_url: str) -> Any:
    """
    Create and return an OpenAI client for a local inference server.
    Works with openai>=1.0.0 which uses ``openai.OpenAI()`` instead of
    the legacy ``openai.ChatCompletion``.
    """
    import openai

    # NOTE: ``api_key`` is required but ignored by local servers
    client = openai.OpenAI(base_url=base_url, api_key="dummy")
    return client


def robust_tool_parse(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool calls from a model message.

    1. Prefer the official ``tool_calls`` field.
    2. Fallback: look for a JSON block inside ``content``.
    """
    # the normal path
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return tool_calls

    # Fallback - scan for a JSON object that looks like a tool call
    content: str = message.get("content", "")
    # Regex which would captures the outermost {...}
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
        # NOTE: the expected output should be: {"name": "...", "arguments": {...}}
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            return [{"id": "fallback", "type": "function", "function": data}]
    except Exception:
        pass

    return []


def format_tool_result(name: str, result: Any) -> str:
    """Human-readable representation of a tool's output for the LLM."""
    return json.dumps({"tool_name": name, "result": result}, ensure_ascii=False)


def _convert_response(raw_msg: Any) -> Dict[str, Any]:
    """
    Convert an OpenAI v1+ Pydantic ChatCompletionMessage object into a
    plain dict so we can pass it to ``robust_tool_parse`` and append it
    to the message history.
    """
    msg: Dict[str, Any] = {
        "role": raw_msg.role if hasattr(raw_msg, "role") else "assistant",
        "content": raw_msg.content if hasattr(raw_msg, "content") else None,
    }

    raw_calls = (
        raw_msg.tool_calls
        if hasattr(raw_msg, "tool_calls") and raw_msg.tool_calls
        else []
    )
    tool_calls_list: List[Dict[str, Any]] = []
    for tc in raw_calls:
        tool_calls_list.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        )
    if tool_calls_list:
        msg["tool_calls"] = tool_calls_list

    return msg


def _process_tool_calls(
    calls: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
) -> None:
    """Execute each tool call and feed the result back into the message list."""
    for call in calls:
        func = call.get("function", {})
        name = func.get("name")
        args_raw = func.get("arguments", {})
        # NOTE: some LLMs return arguments as a JSON string, not a dict
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
        else:
            args = args_raw

        console.print(
            Panel(
                f"[bold]Tool call:[/]\n{name}({json.dumps(args)})",
                title="Tool",
                style="cyan",
            )
        )
        try:
            tool_output = execute_tool(name, args)
        except Exception as exc:
            tool_output = f"Error executing tool {name}: {exc}"
        console.print(Panel(f"[bold]Result:[/]\n{tool_output}", style="green"))
        messages.append(
            {
                "role": "tool",
                "name": name,
                "content": format_tool_result(name, tool_output),
            }
        )


def _call_model(
    client: Any,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_iters: int = 10,
) -> str | None:
    """
    Send *messages* to the model and follow the tool-calling chain until
    a final textual answer arrives (or the iteration limit is hit).

    Returns the final text content, or ``None`` if the limit was reached.
    """
    for _iteration in range(max_iters):
        with console.status("[bold green]Thinking...[/]"):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=1024,
            )
        msg = _convert_response(response.choices[0].message)
        messages.append(msg)

        calls = robust_tool_parse(msg)
        if calls:
            _process_tool_calls(calls, messages)
            continue

        return msg.get("content")

    console.print("[yellow]Reached iteration limit — stopping tool chain.[/]")
    return None


def run_agent_loop(
    client: Any,
    model_name: str,
    system_prompt: str,
) -> None:
    """
    Interactive chat loop.

    Prompts the user for input, sends it to the model, follows the
    tool-calling chain until a text answer arrives, and prints it.
    Loops until the user types *exit* / *quit* or presses Ctrl+C.

    Input conventions:
      · Enter              submit the prompt
      · Alt+Enter          insert a new line (multi-line input)
      · Ctrl+U / Ctrl+K    clear the current input line
      · Ctrl+W              delete the last word
      · Mouse click        position the cursor
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        """Submit the prompt."""
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _(event):
        """Insert a new line (Alt+Enter)."""
        event.current_buffer.insert_text("\n")

    @kb.add("c-u")
    @kb.add("c-k")
    def _(event):
        """Clear the current input line."""
        event.current_buffer.text = ""

    @kb.add("c-w")
    def _(event):
        """Delete the last word."""
        buf = event.current_buffer
        pos = buf.cursor_position
        text = buf.text
        if pos == 0:
            return
        count = 0
        i = pos - 1
        while i >= 0 and text[i] == " ":
            count += 1
            i -= 1
        while i >= 0 and text[i] != " ":
            count += 1
            i -= 1
        buf.delete_before_cursor(count)

    session = PromptSession(
        key_bindings=kb,
        mouse_support=True,
        history=InMemoryHistory(),
    )

    console.print(
        Align.center(
            Panel(
                "[bold green]◆ Agent ready◆ [/]  Type [cyan]Exit[/] or press [cyan]Ctrl+C[/] to quit.\n\n"
                "[dim] [cyan]Enter[/] = Submit | [cyan]Alt+Enter[/] = New Line | [cyan]Ctrl+U/K[/] = Clear | [cyan]Ctrl+W[/] = Delete Word[/]",
                title="[bold white]Session[/]",
                box=box.HEAVY,
                style="green",
            )
        )
    )

    while True:
        try:
            user_input = session.prompt(
                FormattedText([("bold cyan", "You\n> ")]),
            )
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted — exiting.[/]")
            return
        except EOFError:
            console.print("\n[bold red]Goodbye.[/]")
            return

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit", "q"}:
            console.print("[bold red]Goodbye.[/]")
            return

        messages.append({"role": "user", "content": stripped})

        answer = _call_model(client, model_name, messages)
        if answer:
            console.print(Panel("[bold]Assistant[/]", style="magenta"))
            console.print(Markdown(answer))


def main() -> None:
    console.print(Align.center(
        Panel(
            "\n"
            "[bold cyan]::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::[/]\n"
            "[bold cyan]::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::[/]\n"
            "[bold cyan]:::::::::::::::::::::::::::::::%::::::::::::::::::::::::::::::::[/]\n"
            "[bold cyan]::::::::::::::::::::::::::::::%%%=::::::::::::::::::::::::::::::[/]\n"
            "[bold cyan]:::::::::::::::::::::::::::%%%%%*:%%%%::::::::::::::::::::::::::[/]\n"
            "[bold cyan]::::::::::::::::::::::::::::::%:+%::::::::::::::::::::::::::::::[/]\n"
            "[bold cyan]:::::::::::::::::::::::::::::%:::%%%-::::::::::::::::::::::::::[/]\n"
            "[bold cyan]::::::::::::::::::::::::::*%%%%%%%%#%%:::::::::::::::::::::::::[/]\n"
            "[bold cyan]:::::::::::::::::::::::::%%%::::::::%%%:::::::::::::::::::::::::[/]\n"
            "[bold cyan]::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::[/]\n"
            "[bold cyan]::::::::::::::::::::=:::::::::::::::::::::::=:::::::::::::::::::[/]\n"
            "\n"
            "[bold yellow]Terminal-based Agentic Coding Assistant[/]\n"
            "\n"
            "[green]◆[/] Discover local LLMs — [bold]Ollama[/] & [bold]LM Studio[/]\n"
            "[green]◆[/] Interactive tool-calling chat loop\n"
            "[green]◆[/] Web search · File I/O · Command execution\n"
            "\n"
            "[dim]Get started by selecting a model below.[/]\n"
            "\n",
            title="[bold white]Welcome to AgentTool[/]",
            subtitle="[dim]Powered by Ollama & LM Studio[/]",
            box=box.DOUBLE,
            style="bright_blue",
            padding=(1, 2),
        )
    ))

    models = discover_models()
    if not models:
        console.print(
            Panel(
                "No local LLM servers detected.\n"
                "Start Ollama (`ollama serve`) or LM Studio and ensure a model is loaded.",
                style="red",
            )
        )
        sys.exit(1)

    _display_models(models)
    model_name, model_id, base_url = pick_model(models)

    console.print(f"Selected model: [green]{model_name}[/]")
    client = create_openai_client(base_url)

    system_prompt = (
        "You are an expert programming assistant with internet access. "
        "You may call the provided tools to retrieve information, read or write files, "
        "list directories, run shell commands, or perform web searches. "
        "When a tool is called, respond only with the tool invocation; "
        "when you have a final answer, respond in plain English."
    )

    try:
        run_agent_loop(client, model_id, system_prompt)
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user - exiting.[/]")


if __name__ == "__main__":
    main()
