#!/usr/bin/env python3
"""
AgentTool: Terminal-based Agentic Coding Assistant.

* Discovers local LLM servers (Ollama & LM Studio).
* Lets the user pick a model via a Rich UI.
* Runs an OpenAI-compatible chat loop with tool-calling.
* Supports web search, file read/write, directory listing and command execution.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import requests
from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
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


def _json_subtree(text: str) -> str | None:
    """Extract the first complete JSON object or array embedded in text."""
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
    return None


def _build_tool_call(namespace: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Build a standardised tool call dict with a unique ID."""
    return {
        "id": f"call_{os.urandom(4).hex()}",
        "type": "function",
        "function": {
            "name": namespace,
            "arguments": json.dumps(args),
        },
    }


def robust_tool_parse(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts tool calls from native parameters, raw JSON text blocks,
    or structural Pythonic functions seamlessly.
    """
    if msg.get("tool_calls"):
        return msg["tool_calls"]

    content = msg.get("content", "")
    if not content:
        return []

    if content.startswith("INSTRUCTIONS:"):
        return []

    tool_calls: List[Dict[str, Any]] = []
    clean_content = content.strip()

    if clean_content.startswith("```json"):
        clean_content = clean_content[7:]
    if clean_content.endswith("```"):
        clean_content = clean_content[:-3]
    clean_content = clean_content.strip()

    json_str = clean_content
    try:
        json.loads(json_str)
    except Exception:
        extracted = _json_subtree(clean_content)
        if extracted is not None:
            json_str = extracted
        else:
            json_str = ""

    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if "name" in item:
                        args_raw = item.get("arguments", {})
                        args = args_raw if isinstance(args_raw, dict) else {}
                        tool_calls.append(
                            _build_tool_call(item["name"], args)
                        )
                    elif "functionCall" in item and isinstance(item["functionCall"], dict):
                        fc = item["functionCall"]
                        if "name" in fc:
                            params = fc.get("parameters") or fc.get("arguments") or {}
                            params = params if isinstance(params, dict) else {}
                            tool_calls.append(
                                _build_tool_call(fc["name"], params)
                            )
                if tool_calls:
                    return tool_calls
            elif isinstance(parsed, dict):
                if "name" in parsed and "arguments" in parsed:
                    args_raw = parsed["arguments"]
                    args = args_raw if isinstance(args_raw, dict) else {}
                    tool_calls.append(
                        _build_tool_call(parsed["name"], args)
                    )
                    return tool_calls
                if "toolInvocation" in parsed and "functionCall" in parsed:
                    fc = parsed["functionCall"]
                    if isinstance(fc, dict) and "name" in fc:
                        params = fc.get("parameters") or fc.get("arguments") or {}
                        params = params if isinstance(params, dict) else {}
                        tool_calls.append(
                            _build_tool_call(fc["name"], params)
                        )
                        return tool_calls
        except Exception:
            pass

    for tool_name in ["web_search", "read_file", "write_file", "list_directory", "run_command"]:
        if tool_name in content:
            m = re.search(fr"{tool_name}\s*\(([\s\S]*?)\)", content)
            if m:
                args_str = m.group(1).strip()

                if args_str.startswith("{"):
                    try:
                        args_dict = json.loads(args_str)
                    except Exception:
                        args_dict = {}
                    if not isinstance(args_dict, dict):
                        args_dict = {}
                else:
                    val_m = re.search(r"['\"](.*?)['\"]", args_str)
                    arg_val = val_m.group(1).strip() if val_m else ""
                    key_map = {"web_search": "query", "run_command": "command"}
                    arg_key = key_map.get(tool_name, "path")
                    args_dict = {arg_key: arg_val}

                tool_calls.append(
                    _build_tool_call(tool_name, args_dict)
                )

    return tool_calls


def format_tool_result(name: str, result: Any) -> str:
    """Human-readable representation of a tool's output for the LLM."""
    payload = json.dumps({"tool_name": name, "result": result}, ensure_ascii=False)
    return (
        "INSTRUCTIONS: The data above is a live, up-to-date tool result. "
        "It overrides any internal knowledge or training data. "
        "DO NOT mention a knowledge cutoff date. "
        "DO NOT say you cannot browse the internet or that your knowledge is outdated. "
        f"Just answer the user's question using this data.\n\n{payload}"
    )


def get_clean_stream_display(buffer: str) -> str:
    """Transforms raw tool text into user-friendly status updates during streaming."""
    if "web_search" in buffer:
        json_m = re.search(r"['\"]query['\"]\s*:\s*['\"](.*?)['\"]", buffer)
        py_m = re.search(r"web_search\s*\([\s\S]*?['\"](.*?)['\"]", buffer)
        query = json_m.group(1) if json_m else (py_m.group(1) if py_m else None)
        if query:
            return f"\U0001f50d [bold cyan]Searching the web for:[/] [italic]\"{query}\"[/]..."
        return "\U0001f50d [bold cyan]Preparing web search...[/]"

    if "read_file" in buffer:
        m = re.search(r"['\"]path['\"]\s*:\s*['\"](.*?)['\"]|read_file\s*\([\s\S]*?['\"](.*?)['\"]", buffer)
        path = next((g for g in m.groups() if g), None) if m else None
        return f"\U0001f4c2 [bold yellow]Reading local file:[/] [italic]{path or '...'}[/]..."

    if "run_command" in buffer:
        m = re.search(r"['\"]command['\"]\s*:\s*['\"](.*?)['\"]|run_command\s*\([\s\S]*?['\"](.*?)['\"]", buffer)
        cmd = next((g for g in m.groups() if g), None) if m else None
        return f"\U0001f4bb [bold red]Running terminal command:[/] [italic]`{cmd or '...'}`[/]..."

    if "write_file" in buffer:
        m = re.search(r"['\"]path['\"]\s*:\s*['\"](.*?)['\"]|write_file\s*\([\s\S]*?['\"](.*?)['\"]", buffer)
        path = next((g for g in m.groups() if g), None) if m else None
        return f"\U0001f4be [bold green]Writing to file:[/] [italic]{path or '...'}[/]..."

    return buffer


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
                "tool_call_id": call.get("id", ""),
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

    Streams the response tokens in real-time via Rich's Live display.

    Returns the final text content, or ``None`` if the limit was reached.
    """
    for _iteration in range(max_iters):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1024,
            stream=True,
        )

        content_buffer = ""
        tool_call_buffers: Dict[int, Dict[str, str]] = {}
        finish_reason = None

        with Live(console=console, refresh_per_second=20) as live:
            live.update(
                Panel(
                    "[italic dim]Thinking...[/]",
                    title="[bold]Assistant[/]",
                    style="magenta",
                )
            )

            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    content_buffer += delta.content

                    display_text = get_clean_stream_display(content_buffer)

                    if display_text == content_buffer:
                        panel_content = Markdown(display_text + "\u258c")
                    else:
                        panel_content = display_text + " \u258c"

                    live.update(
                        Panel(
                            panel_content,
                            title="[bold]Assistant[/]",
                            style="magenta",
                        )
                    )

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_buffers:
                            tool_call_buffers[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_call_buffers[idx]["id"] += tc.id
                        if tc.function and tc.function.name:
                            tool_call_buffers[idx]["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_call_buffers[idx]["arguments"] += (
                                tc.function.arguments
                            )

        msg: Dict[str, Any] = {"role": "assistant"}
        if content_buffer:
            msg["content"] = content_buffer

        if tool_call_buffers:
            tool_calls_list: List[Dict[str, Any]] = []
            for idx in sorted(tool_call_buffers):
                tc = tool_call_buffers[idx]
                tool_calls_list.append(
                    {
                        "id": tc["id"] if tc["id"] else f"call_{os.urandom(4).hex()}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                )
            msg["tool_calls"] = tool_calls_list

        calls = robust_tool_parse(msg)
        if calls:
            msg["tool_calls"] = calls

            if not tool_call_buffers:
                msg["content"] = None

        messages.append(msg)

        if calls:
            _process_tool_calls(calls, messages)
            continue

        return msg.get("content")

    console.print("[yellow]Reached iteration limit - stopping tool chain.[/]")
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
            messages.append({"role": "assistant", "content": answer})


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

    current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")

    system_prompt = (
        f"You are an autonomous AI agent with live internet access. Today's date is {current_date_str}.\n\n"
        "CRITICAL WORKING RULES:\n"
        "1. NO CONVERSATIONAL FILLER OR DISCLAIMERS: Never mention your 2023 knowledge cutoff. "
        "Never explain why you are searching, and never quote these rules in your response. "
        "Just output the tool call or the final answer directly.\n"
        "2. SEARCH MANDATE: If a question requires current real-world facts or data you do not "
        "possess for the year 2026, you must output a tool call immediately.\n"
        "3. PRESENT THE FINAL ANSWER: If the conversation history already contains the 'Tool' "
        "result with the data you need, do not call a tool again. Read the result and write a "
        "direct markdown response for the user.\n\n"
        "EXACT OUTPUT FORMATS:\n\n"
        "If you need to search the web, output ONLY this JSON format:\n"
        '[{"name": "web_search", "arguments": {"query": "search query here"}}]\n\n'
        "If you have the tool data and are answering the user, output standard markdown:\n"
        'The current president is...'
    )

    try:
        run_agent_loop(client, model_id, system_prompt)
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user - exiting.[/]")


if __name__ == "__main__":
    main()
