"""Ollama (local) LLM integration for PROMETHEON."""

import json
import os
import requests
from safe_executor import execute_command, safe_shutdown
from recycling_bin import trash_file, list_trash, restore

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")

SYSTEM_PROMPT = """You are PROMETHEON — a high-context AI assistant with full system access to PROMETHEUS, a personal NAS. You combine deep technical knowledge with the ability to directly operate the system.

## What You Can Do
You can answer any question — coding, math, science, writing, debugging, architecture, life advice, literally anything. You're like having Claude with terminal access. When a question involves the NAS, you have tools to act directly.

## Tools (use them when relevant)
- `run_command` — Execute system commands on the NAS (ls, cat, df, du, ps, find, grep, sensors, smartctl, git, python, etc.)
- `trash_file` — Move files to recycling bin (auto-purges after 30 days)
- `list_trash` / `restore_from_trash` — Manage recycling bin
- `safe_shutdown` — Safely shut down the NAS

## When to ACT vs ANSWER
- System questions ("check disk health", "what's using space") → use tools, run commands
- Knowledge questions ("explain transformers", "help me debug this code", "write a poem") → just answer directly
- If the user gives a system order, execute it. If they ask a knowledge question, answer it thoroughly.

## Safety Rules
1. NEVER permanently delete files. Always use `trash_file`.
2. The command safety system blocks dangerous commands (rm, dd, mkfs, etc).

## Style
- Be direct and thorough. Give complete, high-quality answers.
- For system operations: act first, explain briefly after.
- For knowledge questions: be as detailed as needed. Use markdown formatting.
- You're a knowledgeable expert and a capable sysadmin rolled into one.
- Use code blocks, bullet points, and structure when it helps clarity.

## System Details
- Debian Linux, Ryzen 3 4300U, commands run locally
- Storage: EVO NVMe 1TB, AirDisk 512GB boot, T7 465GB, T9 1TB, T5 1TB, merged via MergerFS at /srv/mergerfs/PROMETHEUS
- Working directory: /srv/mergerfs/PROMETHEUS
- `sensors` for temps, `smartctl` for drive health, standard Linux tools
- This is Zain's personal NAS. All data is important."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a system command on the NAS (Debian Linux). Only safe commands are allowed (ls, cat, df, du, ps, find, grep, sensors, smartctl, etc.). Destructive commands like rm are blocked. The working directory is /srv/mergerfs/PROMETHEUS (the PROMETHEUS pool).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (e.g., 'ls -la', 'df -h', 'cat /etc/hostname')"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trash_file",
            "description": "Move a file or directory to the PROMETHEON recycling bin instead of permanently deleting it. Items auto-purge after 30 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory to move to trash"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_trash",
            "description": "List all items currently in the PROMETHEON recycling bin, showing original paths, age, and when they'll be purged.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restore_from_trash",
            "description": "Restore an item from the recycling bin to its original location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trash_name": {
                        "type": "string",
                        "description": "The trash name identifier (from list_trash) of the item to restore"
                    }
                },
                "required": ["trash_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "safe_shutdown",
            "description": "Safely shut down the NAS. Syncs all filesystems and schedules power-off in 1 minute. Use when the user says 'shut down', 'power off', 'turn off', or '/shutdown'.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def get_usage_stats():
    """Return model info."""
    return {
        "model": "claude-sonnet-4",
        "local": False,
    }


def _handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if tool_name == "run_command":
        result = execute_command(tool_input["command"])
        if result["blocked"]:
            return f"⛔ {result['stderr']}"
        output = result["stdout"]
        if result["stderr"]:
            output += f"\n[stderr] {result['stderr']}"
        if result["returncode"] != 0:
            output += f"\n[exit code: {result['returncode']}]"
        if not output:
            return "(no output)"
        lines = output.rstrip('\n').split('\n')
        cmd = tool_input.get("command", "")
        if len(lines) > 15:
            truncated = lines[-15:]
            header = f"$ {cmd}\n... ({len(lines) - 15} lines hidden)\n"
            return header + '\n'.join(truncated)
        return f"$ {cmd}\n" + '\n'.join(lines)

    elif tool_name == "trash_file":
        result = trash_file(tool_input["file_path"])
        if result["success"]:
            return f"🗑️ {result['message']} (auto-purges in 30 days)"
        return f"❌ {result['error']}"

    elif tool_name == "list_trash":
        items = list_trash()
        if not items:
            return "Recycling bin is empty."
        lines = ["RECYCLING BIN:", "─" * 60]
        for item in items:
            lines.append(f"  {item['trash_name']}")
            lines.append(f"    Original: {item['original_path']}")
            lines.append(f"    Trashed:  {item['age_str']} | Purges in {item['purge_in']} days")
        return "\n".join(lines)

    elif tool_name == "restore_from_trash":
        result = restore(tool_input["trash_name"])
        if result["success"]:
            return f"✅ {result['message']}"
        return f"❌ {result['error']}"

    elif tool_name == "safe_shutdown":
        result = safe_shutdown()
        if result["success"]:
            steps = " → ".join(result["steps"])
            return f"✅ {steps}\n{result['message']}"
        return f"❌ {result['message']}"

    return f"Unknown tool: {tool_name}"


def _build_messages(conversation_history: list) -> list:
    """Prepend system prompt to conversation history."""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history


def chat(message: str, conversation_history: list, api_key: str = None) -> tuple[str, list]:
    """Send a message to Ollama and handle tool use. Returns (response_text, updated_history)."""
    conversation_history.append({"role": "user", "content": message})

    full_response = ""
    while True:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": _build_messages(conversation_history),
            "tools": TOOLS,
            "stream": False,
        }

        try:
            resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return f"❌ Ollama error: {e}", conversation_history

        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls") or []

        if tool_calls:
            conversation_history.append({
                "role": "assistant",
                "content": msg.get("content", ""),
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result = _handle_tool_call(name, args)
                conversation_history.append({"role": "tool", "content": result})
        else:
            full_response = msg.get("content", "")
            conversation_history.append({"role": "assistant", "content": full_response})
            break

    return full_response.strip(), conversation_history


def chat_stream(message: str, conversation_history: list, api_key: str = None):
    """Stream a response from Ollama, yielding SSE-style event dicts. Handles tool use internally."""
    conversation_history.append({"role": "user", "content": message})

    while True:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": _build_messages(conversation_history),
            "tools": TOOLS,
            "stream": True,
            "options": {"num_ctx": 2048},
        }

        collected_text = ""
        final_msg = {}

        try:
            with requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=90) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        collected_text += content
                        yield {"type": "text", "content": content}
                    if chunk.get("done"):
                        final_msg = msg
                        break
        except requests.exceptions.ReadTimeout:
            yield {"type": "fallback", "reason": "timeout"}
            return
        except requests.RequestException as e:
            yield {"type": "text", "content": f"\n❌ Ollama error: {e}"}
            yield {"type": "done"}
            return

        tool_calls = final_msg.get("tool_calls") or []

        if tool_calls:
            conversation_history.append({
                "role": "assistant",
                "content": collected_text,
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                yield {"type": "tool_call", "name": name, "input": args}
                result = _handle_tool_call(name, args)
                yield {"type": "tool_result", "name": name, "content": result}
                conversation_history.append({"role": "tool", "content": result})
        else:
            conversation_history.append({
                "role": "assistant",
                "content": collected_text or final_msg.get("content", ""),
            })
            break

    yield {"type": "done"}
