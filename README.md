# Anthropic API Proxy for any LLM via LiteLLM 🔄

**Use Anthropic clients (like Claude Code) with any LLM backend.** 🤝

A proxy server that lets you use Anthropic clients with any LLM provider supported by LiteLLM. 🌉

![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- An API key for your chosen LLM provider (e.g., OpenAI, Google, Anthropic, Cohere, etc.) 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

1.  **Clone this repository**:
    ```bash
    git clone https://github.com/1rgs/claude-code-openai.git
    cd claude-code-openai
    ```

2.  **Install uv** (if you haven't already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3.  **Configure Environment Variables**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and fill in your API key and desired model configurations:

    *   `LLM_API_KEY`: The API key for your chosen LLM provider.
    *   `BIG_MODEL` (Optional): The model to map `sonnet` requests to. Defaults to `openai/gpt-4.1`.
    *   `SMALL_MODEL` (Optional): The model to map `haiku` requests to. Defaults to `openai/gpt-4.1-mini`.

4.  **Run the server**:
    ```bash
    uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
    ```
    *(or)*
    ```bash
    claude-proxy --host 0.0.0.0 --port 8082 --reload
    ```
    *(`--reload` is optional, for development)*

### Or simply use the CLI

```bash
uv tool install 'git+https://github.com/ryul99/claude-code-proxy.git'
BIG_MODEL=openai/gpt-4.1 SMALL_MODEL=openai/gpt-4.1 claude-proxy
```

### Using with Claude Code 🎮

1.  **Install Claude Code** (if you haven't already):
    ```bash
    npm install -g @anthropic-ai/claude-code
    ```

2.  **Connect to your proxy**:
    ```bash
    ANTHROPIC_BASE_URL=http://localhost:8082 claude
    ```

    Or (run claude-proxy CLI directly and connect):
    ```bash
    ANTHROPIC_BASE_URL=http://localhost:8082 claude-proxy
    ```

3.  **That's it!** Your Claude Code client will now use the configured backend models through the proxy. 🎯

## Model Mapping 🗺️

The proxy maps Anthropic's `sonnet` and `haiku` models to any LiteLLM-supported model string specified in your `.env` file.

- `Claude Sonnet` -> `BIG_MODEL`
- `Claude Haiku`  -> `SMALL_MODEL`

You **must** use the correct model identifier as required by LiteLLM, which typically includes the provider prefix (e.g., `openai/`, `gemini/`, `anthropic/`).

### Customizing Model Mapping

Control the mapping by setting the environment variables in your `.env` file.
If you use cli command `claude-proxy`, you should set the environment variables in your shell or pass them directly in the command line.

**Example 1: Default (Use OpenAI)**
```dotenv
LLM_API_KEY="your-openai-key"
BIG_MODEL="openai/gpt-4o"
SMALL_MODEL="openai/gpt-4o-mini"
```

**Example 2: Use Google Models**
```dotenv
LLM_API_KEY="your-google-key"
BIG_MODEL="gemini/gemini-1.5-pro-latest"
SMALL_MODEL="gemini/gemini-1.5-flash-latest"
```

**Example 3: Use Anthropic Models**
```dotenv
LLM_API_KEY="your-anthropic-key"
BIG_MODEL="anthropic/claude-3-5-sonnet-20240620"
SMALL_MODEL="anthropic/claude-3-haiku-20240307"
```

For more supported models and provider-specific environment variables (like `AZURE_API_BASE`, etc.), refer to the [LiteLLM documentation](https://docs.litellm.ai/).

## How It Works 🧩

This proxy works by:

1.  **Receiving requests** in Anthropic's API format 📥
2.  **Translating** the requests to a generic format via LiteLLM 🔄
3.  **Sending** the translated request to your configured LLM provider 📤
4.  **Converting** the response back to Anthropic format 🔄
5.  **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁

## Pre-commit Setup 🔧

This project uses pre-commit hooks to ensure code quality. To set up pre-commit hooks, run:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run linting and formatting checks before each commit.
