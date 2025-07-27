# DEPRECATED: Just use litellm proxy instead

litellm proxy now supports Anthropic API Proxy for any LLM via LiteLLM. [docs](https://docs.litellm.ai/docs/tutorials/claude_responses_api)

## Step 1: Install LiteLLM Proxy

```bash
uv tool install 'litellm[proxy]'
# or
pipx install 'litellm[proxy]'
# or
pip install 'litellm[proxy]'
```

## Step 2: Run the Proxy with configuration

```bash
cat <<\EOF >config.yaml
model_list:
  - model_name: "*"
    litellm_params:
      model: "*"
EOF

litellm --config config.yaml
```

## Step 3: Use the Proxy with Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000

claude --model gemini/gemini-2.5-pro
# or
export ANTHROPIC_MODEL=gemini/gemini-2.5-pro
export ANTHROPIC_SMALL_FAST_MODEL=gemini/gemini-2.5-flash
claude
```

# Anthropic API Proxy for any LLM via LiteLLM 🔄

**Use Anthropic clients (like Claude Code) with any LLM backend.** 🤝

A proxy server that lets you use Anthropic clients with any LLM provider supported by LiteLLM. 🌉

![Anthropic API Proxy](pic.png)

## Key Improvements (Compared to original repo) 🚀

- **Universal LLM Support**: Expanded from only supporting OpenAI and Gemini to supporting any LLM provider compatible with LiteLLM.
- **Reliable Tool Calling**: Replaced manual, text handling based tool call handling with OpenAI format based LiteLLM's robust, native translation. This improves the reliability and consistency of tool usage across all models.
- **Fixed thinking budget issues**: The proxy now correctly handles the `thinking_budget` parameter which is enabled by `think hard`, `ultrathink`, etc.
- **Easy Installation and Execution**: Packaged for easy installation via `pip`, `pipx`, or `uv`, and can be run directly with the `claude-proxy` command, eliminating the need for manual cloning and setup.
- **Simplified Configuration**: Replaced the complex `.env` file with straightforward environment variables (`BIG_MODEL`, `SMALL_MODEL`) for easier setup.
- **Improved Code Quality**: The codebase has been refactored for clarity and maintainability, with `pre-commit` hooks integrated to ensure consistent code standards.

## Quick Start ⚡

### Prerequisites

- An API key for your chosen LLM provider (e.g., OpenAI, Google, Anthropic, Cohere, etc.) 🔑

### Setup

Install the proxy server using one of the following methods:

```bash
uv tool install 'git+https://github.com/ryul99/claude-code-proxy.git'
# or
pipx install 'git+https://github.com/ryul99/claude-code-proxy.git'
# or
pip install 'git+https://github.com/ryul99/claude-code-proxy.git'
```

And run the proxy server:

```bash
BIG_MODEL=openai/gpt-4.1 SMALL_MODEL=openai/gpt-4.1 claude-proxy
```

That's it! The proxy server will start listening on port 8082 by default. You can change the port by setting `--port` option

### Using with Claude Code

1.  **Install Claude Code** (if you haven't already):
    ```bash
    npm install -g @anthropic-ai/claude-code
    ```

2.  **Connect to your proxy**:
    ```bash
    ANTHROPIC_BASE_URL=http://localhost:8082 claude
    ```

3.  **That's it!** Your Claude Code client will now use the configured backend models through the proxy. 🎯

## Model Mapping

The proxy maps Anthropic's `sonnet` and `haiku` models to any LiteLLM-supported model string specified in your environment variables.

- `Claude Sonnet` -> `BIG_MODEL`
- `Claude Haiku`  -> `SMALL_MODEL`

You **must** use the correct model identifier as required by LiteLLM, which typically includes the provider prefix (e.g., `openai/`, `gemini/`, `anthropic/`).

### Customizing Model Mapping

Control the mapping by setting the environment variables

**Example 1: Use OpenAI**
```bash
export OPENAI_API_KEY="your-openai-key"
BIG_MODEL="openai/gpt-4o" SMALL_MODEL="openai/gpt-4o-mini" claude-proxy
```

**Example 2: Use Google Models**
```bash
export GEMINI_API_KEY="your-google-key"
BIG_MODEL="gemini/gemini-2.5-pro" SMALL_MODEL="gemini/gemini-2.5-flash" claude-proxy
```

**Example 3: Use Anthropic Models**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
BIG_MODEL="anthropic/claude-sonnet-4-20250514" SMALL_MODEL="anthropic/claude-3-haiku-20240307"
```

API keys for the respective providers must be set in your environment variables. The proxy will automatically use these keys to authenticate requests to the LLM provider.

This means you can use different providers for `BIG_MODEL` and `SMALL_MODEL` if desired, allowing for flexible configurations based on your needs.

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
