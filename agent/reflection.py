"""
Strict JSON parser for LLM reflection output.

Enforces the reflection JSON contract:
  {"critique": str, "decision": "continue"|"search_again"|"stop", "store_memory": bool}

No regex fallbacks, no ast.literal_eval. Valid JSON or parse_error.
"""

import json

# ---------------------------------------------------------------------------
# Allowed schema
# ---------------------------------------------------------------------------
_REFLECTION_KEYS = {"critique", "decision", "store_memory"}
_VALID_DECISIONS = {"continue", "search_again", "stop"}


def strict_reflection_parse(raw: str) -> dict:
    """
    Parse raw LLM reflection output into a validated dict.

    Rules:
      - Must be valid JSON
      - Must contain 'critique' (str), 'decision' (str), 'store_memory' (bool)
      - decision must be one of: continue, search_again, stop
      - No extra keys allowed
      - Any violation returns {"parse_error": "<reason>"}
    """
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        raw = "\n".join(lines).strip()

    # Find first { and last matching }
    brace_start = raw.find("{")
    if brace_start == -1:
        return {"parse_error": "No JSON object found in reflection output"}

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
        return {"parse_error": "Unbalanced braces in reflection output"}

    json_str = raw[brace_start:brace_end + 1]

    # Parse JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"parse_error": f"Invalid JSON in reflection: {e}"}

    if not isinstance(parsed, dict):
        return {"parse_error": "Reflection JSON root must be an object"}

    # Validate required fields
    if "critique" not in parsed:
        return {"parse_error": "Missing required field: 'critique'"}
    if not isinstance(parsed["critique"], str):
        return {"parse_error": "'critique' must be a string"}

    if "decision" not in parsed:
        return {"parse_error": "Missing required field: 'decision'"}
    if parsed["decision"] not in _VALID_DECISIONS:
        return {
            "parse_error": f"Invalid decision: '{parsed['decision']}'. "
                           f"Must be one of {_VALID_DECISIONS}"
        }

    if "store_memory" not in parsed:
        return {"parse_error": "Missing required field: 'store_memory'"}
    if not isinstance(parsed["store_memory"], bool):
        return {"parse_error": "'store_memory' must be a boolean"}

    # Check for extra keys
    extra = set(parsed.keys()) - _REFLECTION_KEYS
    if extra:
        return {"parse_error": f"Unexpected keys in reflection: {extra}"}

    return parsed
