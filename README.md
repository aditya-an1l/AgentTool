<div align="center">
  <img src="./media/logo.jpeg" width="250" height="250" alt="AgentTool Logo">
</div>
<div align="center">
  <h1>AgentTool</h1>
  <p>Terminal-based agentic coding assistant powered by local LLMs.</p>
</div>

<div align="center">
  <p>
    <a href="https://github.com/aditya-an1l/AgentTool/pulse">
      <img alt="Last commit" src="https://img.shields.io/github/last-commit/aditya-an1l/AgentTool?style=for-the-badge&logo=git&color=1DA1F2&logoColor=FFFFFF&labelColor=000000"/>
    </a>
<a href="https://github.com/aditya-an1l/AgentTool/blob/main/LICENSE">
  <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge&logo=apache" />
</a>
    <a href="https://github.com/aditya-an1l/AgentTool/stargazers">
      <img alt="Stars" src="https://img.shields.io/github/stars/aditya-an1l/AgentTool?style=for-the-badge&logo=starship&color=1DA1F2&logoColor=FFFFFF&labelColor=000000" />
    </a>
    <a href="https://github.com/aditya-an1l/AgentTool/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/aditya-an1l/AgentTool?style=for-the-badge&logo=gitbook&color=FF4136&logoColor=FFFFFF&labelColor=000000" />
    </a>
    <a href="https://github.com/aditya-an1l/AgentTool">
      <img alt="Repo Size" src="https://img.shields.io/github/repo-size/aditya-an1l/AgentTool?color=1DA1F2&label=SIZE&logo=files&style=for-the-badge&logoColor=FFFFFF&labelColor=000000" />
    </a>
    <a href="https://twitter.com/intent/follow?screen_name=aditya_an1l">
      <img alt="follow on X" src="https://img.shields.io/twitter/follow/aditya_an1l?style=for-the-badge&logo=x&color=1DA1F2&logoColor=FFFFFF&labelColor=000000" />
    </a>
  </p>
</div>

## Overview
<div align="center">
<img width="818" height="624" alt="image" src="https://github.com/user-attachments/assets/c8fe424e-00ca-472e-a194-1dfc18cdf4ee" />
</div>

AgentTool is a lightweight Python CLI that discovers locally-running LLM servers (Ollama or LM Studio), lets you pick a model, and then runs an OpenAI-compatible **agentic loop** with tool-calling capabilities. It acts as a meta-agent: it auto-discovers models, provides an interactive selection menu, and powers an agentic tool-calling loop with web search, file operations, and shell command execution.

## Features

<div align="center">
<img width="1913" height="520" alt="image" src="https://github.com/user-attachments/assets/99aa7878-f405-4955-b78c-5a459c142852" />
</div>


- **Auto-detect** Ollama (`http://localhost:11434`) and LM Studio (`http://localhost:1234`) models.
- Interactive selection UI powered by **Rich**.
- Supports tools:
  - `web_search` (DuckDuckGo)
  - `read_file` / `write_file`
  - `list_directory`
  - `run_command`
- Robust fallback parsing for models that forget the proper `tool_calls` JSON format.
- Graceful `Ctrl+C` handling.
- Comprehensive test suite.

## Installation

```bash
# Clone the repository:
git clone https://github.com/aditya-an1l/AgentTool.git
cd AgentTool

# Set up a virtual environment:
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Install dev dependencies (for running tests):
pip install -r requirements-dev.txt
```

## Usage

1. **Start a local inference server**
   - **Ollama**: `ollama serve` (ensure a model is installed, e.g., `ollama pull llama3.1`)
   - **LM Studio**: Open the LM Studio UI and load a model.

2. **Run the assistant:**

```bash
python agent.py
```

Select a model by number, then converse with the assistant. The assistant can call the built-in tools to read/write files, search the web, list directories, or execute shell commands.

## Running Tests

```bash
# Install dev dependencies first, then:
python -m pytest tests/ -v
```

## Notes

- No real API key is required; a dummy key is supplied internally.
- The script works with any OpenAI-compatible local endpoint (Ollama, LM Studio, etc.).
- The maximum loop depth is limited to 10 iterations to avoid endless cycles.

<!--
# issues that needds to be resolved:
# 1. Escape sequences for arrow keys are not handled properly in the prompt. 
# For example, when the user presses the left or right arrow key, it produces 
# characters like ^[[D and ^[[C instead of moving the cursor.
#
# 2. The prompt does not support multi-line input. If the user presses Enter, the input is submitted immediately, 
# and there is no way to enter a multi-line prompt. A better approach would be to allow the user to have multi-line input, 
# perhaps by using a different key combination to add a new line by the input  Shift+Enter, and then use Enter key to have
# the prompt being submitted. In the run_agent_loop function,
# mention that the user can use Shift+Enter for new line, and Enter for multi-line input.
#
# 3. Have the following keymaps
#   - Ctrl+U or Ctrl+K to clear the current input line.
#   - Ctrl+W or Ctrl+Backspace to delete the last word.
#
# 4. Mouse cursor support 
# The user must be able to go to a specific part of the prompt by clicking on the any character in the prompt.
-->
