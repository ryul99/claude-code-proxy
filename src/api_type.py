import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, field_validator

BIG_MODEL = os.environ.get("BIG_MODEL", "openai/gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "openai/gpt-4.1-mini")

logger = logging.getLogger("rich")


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    type: Optional[str] = None
    budget_tokens: Optional[int] = None


class TokenCountResponse(BaseModel):
    input_tokens: int


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator("model")
    @classmethod
    def validate_model_token_count(cls, v, info):  # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v  # Default to original value

        logger.debug(
            f"TOKEN COUNT VALIDATION: Original='{original_model}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'"
        )

        clean_v = v

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Haiku to SMALL_MODEL based on provider preference
        if "haiku" in clean_v.lower():
            new_model = SMALL_MODEL
            mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif "sonnet" in clean_v.lower():
            new_model = BIG_MODEL
            mapped = True

        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"TOKEN COUNT MAPPING: '{original_model}' -> '{new_model}'")
        else:
            new_model = v  # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model

        return new_model


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator("model")
    @classmethod
    def validate_model_field(cls, v, info):  # Renamed to avoid conflict
        original_model = v
        new_model = v  # Default to original value

        logger.debug(
            f"MODEL VALIDATION: Original='{original_model}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'"
        )

        clean_v = v

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Haiku to SMALL_MODEL based on provider preference
        if "haiku" in clean_v.lower():
            new_model = SMALL_MODEL
            mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif "sonnet" in clean_v.lower():
            new_model = BIG_MODEL
            mapped = True

        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"MODEL MAPPING: '{original_model}' -> '{new_model}'")
        else:
            new_model = v  # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model

        return new_model
