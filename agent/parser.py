"""
Strict JSON parser for LLM output.

Enforces the JSON output contract:
  - Tool action:   {"thought", "action_type": "tool", "tool_name", "tool_input"}
  - Final answer:  {"thought", "action_type": "final_answer", "final_answer"}

No regex fallbacks, no ast.literal_eval. Valid JSON or parse_error.
"""

import json

# ---------------------------------------------------------------------------
# Allowed schemas — used for key validation
# ---------------------------------------------------------------------------
_TOOL_KEYS = {"thought", "action_type", "tool_name", "tool_input"}
_FINAL_KEYS = {"thought", "action_type", "final_answer"}
_VALID_ACTION_TYPES = {"tool", "final_answer"}


def strict_json_parse(raw: str) -> dict:
    """
    Parse raw LLM output into a validated dict.

    Rules:
      - Must be valid JSON (no ast.literal_eval, no regex fallback)
      - Must contain 'thought' (str) and 'action_type' (str)
      - action_type == 'tool'         → requires 'tool_name' (str), 'tool_input' (dict)
      - action_type == 'final_answer' → requires 'final_answer' (str)
      - No extra keys allowed
      - Any violation returns {"parse_error": "<reason>"}
    """
    # --- Step 1: extract JSON object from raw text ---
    raw = raw.strip()

    # Strip markdown code fences if present (common LLM decoration)
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        raw = "\n".join(lines).strip()

    # Find first { and last matching }
    brace_start = raw.find("{")
    if brace_start == -1:
        return {"parse_error": "No JSON object found in LLM output"}

    brace_count = 0
    brace_end = -1
    for i in range(brace_start, len(raw)):
        if raw[i] == "{":
            brace_count += 1
        elif raw[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break

    if brace_end == -1:
        return {"parse_error": "Unbalanced braces in LLM output"}

    json_str = raw[brace_start:brace_end + 1]

    # --- Step 2: parse JSON ---
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"parse_error": f"Invalid JSON: {e}"}

    if not isinstance(parsed, dict):
        return {"parse_error": "JSON root must be an object"}

    # --- Step 3: validate common required fields ---
    if "thought" not in parsed:
        return {"parse_error": "Missing required field: 'thought'"}
    if not isinstance(parsed["thought"], str):
        return {"parse_error": "'thought' must be a string"}

    if "action_type" not in parsed:
        return {"parse_error": "Missing required field: 'action_type'"}
    if parsed["action_type"] not in _VALID_ACTION_TYPES:
        return {
            "parse_error": f"Invalid action_type: '{parsed['action_type']}'. "
                           f"Must be one of {_VALID_ACTION_TYPES}"
        }

    # --- Step 4: validate per action_type ---
    action_type = parsed["action_type"]

    if action_type == "tool":
        # Check required tool-specific fields
        if "tool_name" not in parsed:
            return {"parse_error": "action_type 'tool' requires 'tool_name'"}
        if not isinstance(parsed["tool_name"], str):
            return {"parse_error": "'tool_name' must be a string"}
        if "tool_input" not in parsed:
            return {"parse_error": "action_type 'tool' requires 'tool_input'"}
        if not isinstance(parsed["tool_input"], dict):
            return {"parse_error": "'tool_input' must be a JSON object (dict)"}

        # Check for extra keys
        extra = set(parsed.keys()) - _TOOL_KEYS
        if extra:
            return {"parse_error": f"Unexpected keys for tool action: {extra}"}

    elif action_type == "final_answer":
        if "final_answer" not in parsed:
            return {"parse_error": "action_type 'final_answer' requires 'final_answer'"}
        if not isinstance(parsed["final_answer"], str):
            return {"parse_error": "'final_answer' must be a string"}

        extra = set(parsed.keys()) - _FINAL_KEYS
        if extra:
            return {"parse_error": f"Unexpected keys for final_answer: {extra}"}

    return parsed