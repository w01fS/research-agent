import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedOutput:
    thought: str
    action: Optional[dict] = None
    final: Optional[str] = None
    error: Optional[str] = None


class OutputParser:

    def parse(self, raw: str) -> ParsedOutput:
        thought = self._extract_section(raw, "THOUGHT") or ""
        action_block = self._extract_json_block(raw)
        final = self._extract_section(raw, "FINAL")

        if action_block and final:
            return ParsedOutput(thought=thought, error="Both ACTION and FINAL present")

        if not action_block and not final:
            return ParsedOutput(thought=thought, error="Neither ACTION nor FINAL present")

        if action_block:
            try:
                action_json = json.loads(action_block)
            except json.JSONDecodeError:
                # Fallback: handle single quotes (common LLM error)
                import ast
                try:
                    action_json = ast.literal_eval(action_block)
                    if not isinstance(action_json, dict):
                         return ParsedOutput(thought=thought, error="Malformed ACTION JSON")
                except (ValueError, SyntaxError):
                    return ParsedOutput(thought=thought, error="Malformed ACTION JSON")

            # 🔒 Schema validation
            if "tool" not in action_json:
                return ParsedOutput(thought=thought, error="ACTION missing 'tool' field")

            if "input" not in action_json:
                return ParsedOutput(thought=thought, error="ACTION missing 'input' field")

            if not isinstance(action_json["input"], dict):
                return ParsedOutput(thought=thought, error="'input' must be a dict")

            return ParsedOutput(thought=thought, action=action_json)

        return ParsedOutput(thought=thought, final=(final or "").strip())

    def _extract_section(self, raw: str, section: str) -> Optional[str]:
        # Try with colon first (strict match)
        pattern = rf"^{section}:\s*(.*?)(?=\n[A-Z]+[:\s]|\Z)"
        match = re.search(pattern, raw, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip() or None

        # Fallback: match without colon (e.g. "FINAL some answer")
        pattern_no_colon = rf"^{section}\s+(.*?)(?=\n[A-Z]+[:\s]|\Z)"
        match = re.search(pattern_no_colon, raw, re.DOTALL | re.MULTILINE)
        return match.group(1).strip() if match else None

    def _extract_json_block(self, raw: str) -> Optional[str]:
        action_start = raw.find("ACTION:")
        if action_start == -1:
            # Fallback: try matching "ACTION" without colon
            action_match = re.search(r"^ACTION\s", raw, re.MULTILINE)
            if action_match:
                action_start = action_match.start()
            else:
                return None

        brace_start = raw.find("{", action_start)
        if brace_start == -1:
            return None

        brace_count = 0
        for i in range(brace_start, len(raw)):
            if raw[i] == "{":
                brace_count += 1
            elif raw[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return raw[brace_start:i+1]

        return None