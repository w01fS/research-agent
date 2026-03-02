import re

class ParseError(Exception):
    pass

def parse_llm_output(text: str):
    thought_match = re.search(r"THOUGHT:(.*)", text)
    action_match = re.search(r"ACTION:(.*)", text)
    final_match = re.search(r"FINAL:(.*)", text)
    observation_match = re.search(r"OBSERVATION:(.*)", text)

    if not thought_match:
        raise ParseError("Missing THOUGHT")

    if observation_match:
        raise ParseError("LLM attempted to inject OBSERVATION")

    if action_match and final_match:
        raise ParseError("Both ACTION and FINAL present")

    if not action_match and not final_match:
        raise ParseError("Neither ACTION nor FINAL present")

    thought = thought_match.group(1).strip()

    return {
        "thought": thought,
        "action": action_match.group(1).strip() if action_match else None,
        "final": final_match.group(1).strip() if final_match else None
    }