# AI Bash Agent

A Python-based AI agent that interacts with users through natural language and executes bash commands on their behalf. The agent supports multiple LLM backends (llama.cpp and OpenAI/ChatGPT) and uses structured JSON responses with strict schema validation for safe command execution.

## Features

- **Multi-backend support**: Works with llama.cpp servers and OpenAI/ChatGPT API
- **Structured JSON responses**: Strict schema ensures reliable parsing of agent responses
- **Safe command execution**: Commands run in isolated temporary directories with safety checks
- **File operations**: Support for writing files and applying patches within the workspace
- **Interactive CLI**: Conversational interface with commands like `clear`, `status`, and `exit`
- **Configurable parameters**: Customizable model, temperature, token limits, and confirmation prompts

## Installation

### Prerequisites

- Python 3.8+
- Bash
- `patch` utility (for applying code patches)

### Setup

1. Clone or copy the project files to your desired location:
   ```bash
   cd /home/intl/wrk/dlab/ai-bash-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install requests openai
   ```

4. Configure environment variables (optional):
   - Copy `.env.example` to `.env` and set your OpenAI API key
   - The agent will use `OPENAI_API_KEY` from environment when using OpenAI backend

## Usage

### Basic Run

```bash
python agent.py
```

### Command-line Options

| Option | Description |
|--------|-------------|
| `--server` | Backend to use: `llama.cpp`, `chatgpt`, or `openai` (default: `llama.cpp`) |
| `-u`, `--url` | URL for llama.cpp server (default: `http://192.168.3.11:8080/v1/chat/completions`) |
| `-m`, `--model` | Model name/id (defaults: `qwen` for llama.cpp, `gpt-5.4` for OpenAI) |
| `--api-key` | API key for llama.cpp-compatible servers |
| `--max-tokens` | Maximum tokens for completion (default: 4096) |
| `--temp` | Temperature for generation (default: 0.7) |
| `--no-confirm` | Execute commands without confirmation (DANGER) |
| `--no-json-schema` | Disable JSON schema validation |

### Examples

```bash
# Run with llama.cpp server
python agent.py --server llama.cpp --url http://localhost:8080/v1/chat/completions --model my-model

# Run with OpenAI backend
python agent.py --server chatgpt --model gpt-5

# Run without confirmation (use with caution)
python agent.py --server llama.cpp --no-confirm
```

### Interactive Commands

- Type your request normally
- `exit` - Quit the agent
- `clear` - Reset conversation history and workspace
- `status` - Show current configuration and workspace information

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required for OpenAI backend)
- `OPENAI_BASE_URL` - Custom OpenAI-compatible base URL (optional)

### Backend Setup

#### llama.cpp

1. Start a llama.cpp server with your preferred model
2. Use the default URL or specify with `--url`
3. Model name defaults to `qwen` but can be customized

#### OpenAI/ChatGPT

1. Set `OPENAI_API_KEY` environment variable
2. Use `--server chatgpt` or `--server openai`
3. Model defaults to `gpt-5.4` but can be customized

## Technical Details

### Agent Response Schema

The agent responds with strictly structured JSON objects containing:

- `type`: Either `"message"` or `"tool"`
- `plan`: Array of 1-3 short bullet points explaining the approach
- `message`: User-facing response text
- `files`: Optional array of files to write (with `path` and `content`)
- `patches`: Optional array of patches to apply (with `path` and `diff`)
- `tool`: For tool type, specifies the tool to use (currently only `bash`)
- `commands`: For tool type, array of bash commands to execute

### Safety Features

- Commands run in isolated temporary directories
- File paths are validated to prevent escaping the workspace
- Patches are validated to prevent referencing other files
- Confirmation prompts before executing commands (unless `--no-confirm` is used)
- Timeout protection for long-running commands (default: 300 seconds)
- Maximum tool round limit (default: 32) to prevent infinite loops

### Workspace Management

- Each session gets a unique temporary directory
- Files written by the agent are stored in this workspace
- The workspace is cleaned up when the session ends
- The `clear` command creates a new workspace while preserving conversation history

## Contributing

This is a research/development project. Feel free to modify and extend the code as needed.

## License

This project is provided as-is for educational and research purposes.