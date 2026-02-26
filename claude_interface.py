"""Anthropic Claude API integration for PROMETHEON (vision + tools)."""

import json
import os

from anthropic import Anthropic
from llm_interface import _handle_tool_call, SYSTEM_PROMPT

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Tools in Anthropic format (input_schema instead of parameters)
CLAUDE_TOOLS = [
    {
        "name": "run_command",
        "description": "Execute a system command on the NAS (Debian Linux). Only safe commands are allowed (ls, cat, df, du, ps, find, grep, sensors, smartctl, etc.). Destructive commands like rm are blocked. The working directory is /srv/mergerfs/PROMETHEUS (the PROMETHEUS pool).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (e.g., 'ls -la', 'df -h', 'cat /etc/hostname')"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "trash_file",
        "description": "Move a file or directory to the PROMETHEON recycling bin instead of permanently deleting it. Items auto-purge after 30 days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to move to trash"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "list_trash",
        "description": "List all items currently in the PROMETHEON recycling bin, showing original paths, age, and when they'll be purged.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "restore_from_trash",
        "description": "Restore an item from the recycling bin to its original location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "trash_name": {
                    "type": "string",
                    "description": "The trash name identifier (from list_trash) of the item to restore"
                }
            },
            "required": ["trash_name"]
        }
    },
    {
        "name": "safe_shutdown",
        "description": "Safely shut down the NAS. Syncs all filesystems and schedules power-off in 1 minute. Use when the user says 'shut down', 'power off', 'turn off', or '/shutdown'.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]


def _content_block_to_dict(block):
    """Convert an Anthropic SDK content block object to a plain dict with only API-accepted fields."""
    if isinstance(block, dict):
        btype = block.get("type")
        if btype == "text":
            return {"type": "text", "text": block["text"]}
        if btype == "tool_use":
            return {"type": "tool_use", "id": block["id"], "name": block["name"], "input": block["input"]}
        return block
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return block


def chat_stream(message: str, conversation_history: list, api_key: str,
                image_b64: str = None, image_mime: str = None):
    """Stream a response from Claude API, yielding SSE-style event dicts.

    Handles tool use in a loop (like llm_interface.chat_stream).
    Accepts an optional base64 image for vision turns.
    """
    client = Anthropic(api_key=api_key)

    # Build initial user content
    if image_b64:
        user_content = []
        if message:
            user_content.append({"type": "text", "text": message})
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_mime or "image/jpeg",
                "data": image_b64,
            }
        })
    else:
        user_content = message

    conversation_history.append({"role": "user", "content": user_content})

    while True:
        collected_text = ""

        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=16384,
                system=SYSTEM_PROMPT,
                tools=CLAUDE_TOOLS,
                messages=conversation_history,
            ) as stream:
                for text_chunk in stream.text_stream:
                    collected_text += text_chunk
                    yield {"type": "text", "content": text_chunk}

                final_message = stream.get_final_message()

        except Exception as e:
            yield {"type": "text", "content": f"\n❌ Claude API error: {e}"}
            yield {"type": "done"}
            return

        # Convert SDK content blocks → plain dicts for JSON serializability
        serializable_content = [_content_block_to_dict(b) for b in final_message.content]
        conversation_history.append({
            "role": "assistant",
            "content": serializable_content,
        })

        # Check for tool use (stop_reason == "tool_use")
        tool_uses = [b for b in final_message.content if b.type == "tool_use"]
        if not tool_uses:
            break

        # Execute each tool call and collect results
        tool_results = []
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            yield {"type": "tool_call", "name": tool_name, "input": tool_input}
            result = _handle_tool_call(tool_name, tool_input)
            yield {"type": "tool_result", "name": tool_name, "content": result}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        # Feed tool results back so the model can continue
        conversation_history.append({"role": "user", "content": tool_results})

    yield {"type": "done"}
