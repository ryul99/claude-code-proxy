import json
import logging
import uuid
from typing import Any, Dict, Union

from src.api_type import MessagesRequest, MessagesResponse, Usage

logger = logging.getLogger("rich")


def _stringify_tool_content(tool_content: Any) -> str:
    if tool_content is None:
        return "None"
    if isinstance(tool_content, str):
        return tool_content
    if isinstance(tool_content, list):
        text = ""
        for item in tool_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "")
            else:
                try:
                    text += json.dumps(item)
                except:
                    text += str(item)
        return text
    try:
        return json.dumps(tool_content)
    except:
        return str(tool_content)


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format

    messages = []

    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, "type") and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"

            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})

    # Add conversation messages
    for msg in anthropic_request.messages:
        content = msg.content

        # Handle simple string content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        # --- Process Assistant Messages ---
        if msg.role == "assistant":
            text_content = ""
            tool_calls = []
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_content += block.text + "\n"
                    elif block.type == "tool_use":
                        # Convert Anthropic tool_use to OpenAI tool_calls
                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input)
                                    if isinstance(block.input, dict)
                                    else str(block.input),
                                },
                            }
                        )

            assistant_message = {
                "role": "assistant",
                "content": text_content.strip() or None,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            # OpenAI requires content to be a string, even if empty, when tool_calls are present.
            if not assistant_message.get("content") and assistant_message.get(
                "tool_calls"
            ):
                assistant_message["content"] = ""

            messages.append(assistant_message)

        # --- Process User Messages ---
        elif msg.role == "user":
            user_text_content = ""
            # First, collect all text blocks into one user message
            for block in content:
                if hasattr(block, "type") and block.type == "text":
                    user_text_content += block.text + "\n"

            if user_text_content.strip():
                messages.append({"role": "user", "content": user_text_content.strip()})

            # Second, create 'tool' messages for each tool_result
            for block in content:
                if hasattr(block, "type") and block.type == "tool_result":
                    result_content_str = _stringify_tool_content(
                        getattr(block, "content", "")
                    )
                    tool_call_id = getattr(block, "tool_use_id", "")

                    if tool_call_id:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": result_content_str,
                            }
                        )

        # --- Fallback for other roles ---
        else:
            try:
                messages.append(msg.dict())
            except:
                messages.append(msg)

    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    # if not anthropic_request.model.startswith("anthropic/"):
    #     max_tokens = min(max_tokens, 16384)
    #     logger.debug(
    #         f"Capping max_tokens to 16384 for non-Anthropic model (original value: {anthropic_request.max_tokens})"
    #     )

    # Convert thinking configuration to OpenAI format if present
    reasoning_effort = None
    if anthropic_request.thinking and anthropic_request.thinking.type == "enabled":
        if anthropic_request.thinking.budget_tokens <= 1024:
            reasoning_effort = "low"
        elif anthropic_request.thinking.budget_tokens <= 2048:
            reasoning_effort = "medium"
        else:
            reasoning_effort = "high"
        logger.debug(
            f"Converted thinking configuration {anthropic_request.thinking} to reasoning_effort: {reasoning_effort}"
        )

    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Add optional parameters if present
    if reasoning_effort:
        litellm_request["reasoning_effort"] = reasoning_effort

    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences

    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p

    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k

    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, "dict"):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                    logger.error(f"Could not convert tool to dict: {tool}")
                    continue  # Skip this tool if conversion fails

            input_schema = tool_dict.get("input_schema", {})

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema,  # Use potentially cleaned schema
                },
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools

    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, "dict"):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice

        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]},
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"

    return litellm_request


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest
) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""

    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = (
                message.content if message and hasattr(message, "content") else ""
            )
            tool_calls = (
                message.tool_calls
                if message and hasattr(message, "tool_calls")
                else None
            )
            finish_reason = (
                choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            )
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = (
                    litellm_response
                    if isinstance(litellm_response, dict)
                    else litellm_response.dict()
                )
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__
                try:
                    response_dict = (
                        litellm_response.model_dump()
                        if hasattr(litellm_response, "model_dump")
                        else litellm_response.__dict__
                    )
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, "id", f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, "choices", [{}]),
                        "usage": getattr(litellm_response, "usage", {}),
                    }

            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = (
                choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            )
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = (
                choices[0].get("finish_reason", "stop")
                if choices and len(choices) > 0
                else "stop"
            )
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        # Create content list for Anthropic format
        content = []

        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")

            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = (
                        getattr(function, "arguments", "{}") if function else "{}"
                    )

                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool arguments as JSON: {arguments}"
                        )
                        arguments = {"raw": arguments}

                logger.debug(
                    f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}"
                )

                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": arguments,
                    }
                )

        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default

        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})

        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

        return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        )
        logger.error(error_message)

        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": f"Error converting response: {str(e)}. Please check server logs.",
                }
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )
