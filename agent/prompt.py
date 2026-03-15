"""
Prompt builders for the research agent.

Two prompt functions:
  - build_think_prompt(state, registry, memories) → str
  - build_reflect_prompt(state) → str
"""

import json
from agent.state import AgentState
from tools.registry import ToolRegistry

# Maximum number of recent reasoning steps to include in context
_MAX_RECENT_STEPS = 3


def build_think_prompt(state: AgentState, registry: ToolRegistry, memories: list) -> str:
    """
    Build the LLM prompt for the THINK phase.

    Context structure:
      1. Question
      2. Relevant Memories (from FAISS)
      3. Recent Reasoning Steps (capped)
      4. Current Observation (if applicable)
      5. Output contract
    """
    # --- Tool descriptions ---
    tools_desc = ""
    for name, tool in registry.list_tools().items():
        tools_desc += f'  - "{name}": {tool.description}\n'

    # --- Relevant memories ---
    if memories:
        memory_lines = ""
        for i, mem in enumerate(memories, 1):
            memory_lines += f"  [{i}] {mem}\n"
        memory_block = (
            f"YOU ALREADY KNOW THESE FACTS FROM PREVIOUS RESEARCH:\n"
            f"{memory_lines}\n"
            f"IMPORTANT: You already have information above. If it answers the question, "
            f"you MUST use Format B (final_answer) now. Do NOT search for information you already have."
        )
    else:
        memory_block = "KNOWN FACTS: (none yet — use tools to research)"

    # --- Recent reasoning steps (capped to avoid context explosion) ---
    history = ""
    start = max(0, len(state.thoughts) - _MAX_RECENT_STEPS)
    for i in range(start, len(state.thoughts)):
        history += f"THOUGHT: {state.thoughts[i]}\n"
        if i < len(state.actions):
            action_json = json.dumps(state.actions[i])
            history += f"ACTION: {action_json}\n"
        if i < len(state.observations):
            history += f"OBSERVATION: {state.observations[i]}\n"
        if i < len(state.reflections):
            history += f"REFLECTION: {state.reflections[i]}\n"
        history += "\n"

    return f"""You are a research agent. Answer questions by using tools step-by-step.

YOUR OUTPUT MUST BE A SINGLE RAW JSON OBJECT. No markdown, no explanation, no extra text.

Available tools:
{tools_desc}

Choose ONE of these two JSON formats:

Format A — to call a tool (ONLY if you need NEW information):
{{"thought": "<your reasoning>", "action_type": "tool", "tool_name": "<tool>", "tool_input": {{"<key>": "<value>"}}}}

Example (search_local_knowledge):
{{"thought": "I need to search for semiconductor info.", "action_type": "tool", "tool_name": "search_local_knowledge", "tool_input": {{"query": "semiconductor"}}}}

Example (calculator):
{{"thought": "I need to calculate 2 + 2.", "action_type": "tool", "tool_name": "calculator", "tool_input": {{"expression": "2+2"}}}}

Format B — to give a final answer (USE THIS if you already have the information):
{{"thought": "<your reasoning>", "action_type": "final_answer", "final_answer": "<your answer>"}}

Example (final answer):
{{"thought": "I have found all necessary information.", "action_type": "final_answer", "final_answer": "The roadblocks include geopolitical tensions and raw material shortages."}}

RULES:
1. Output ONLY the JSON object. No markdown fences.
2. action_type must be exactly "tool" or "final_answer".
3. tool_name must be one of the tools listed above.
4. tool_input must be a JSON object (not a string).
5. final_answer MUST be a string. Do NOT use a JSON object or list for final_answer.
6. Do NOT invent fields. Use ONLY the keys shown above.
7. You MUST always include the "thought" field in your response.

RECENT REASONING STEPS:
{history if history else "(none yet)"}

Question:
{state.question}

{memory_block}

STOPPING CRITERIA:
- If KNOWN FACTS above already answer the question, you MUST respond with Format B immediately.
- DO NOT call a tool to retrieve information you already have in KNOWN FACTS.
- DO NOT repeat the same tool call with the same input.

Iteration: {state.iteration}/{state.max_iterations}

Respond with a single JSON object (ONLY the JSON, must include "thought") now:"""


def build_reflect_prompt(state: AgentState) -> str:
    """
    Build the LLM prompt for the REFLECT phase.

    Asks the LLM to critique the latest reasoning step and observation,
    then decide whether to continue, search again, or stop.
    """
    latest_thought = state.thoughts[-1] if state.thoughts else "(none)"
    latest_action = json.dumps(state.actions[-1]) if state.actions else "(none)"
    latest_observation = state.observations[-1] if state.observations else "(none)"

    return f"""You are a research agent performing self-reflection.

Review the latest reasoning step and observation, then critique the direction of the research.

YOUR OUTPUT MUST BE A SINGLE RAW JSON OBJECT. No markdown, no explanation, no extra text.

Output format:
{{"critique": "<your analysis of the last step>", "decision": "<continue|search_again|stop>", "store_memory": <true|false>}}

Fields:
- critique: Your analysis of whether the last reasoning step and observation moved closer to answering the question. Be specific.
- decision: One of:
  - "continue" — the current direction is productive, keep investigating
  - "search_again" — the observation was not useful, try a different search angle
  - "stop" — we have enough information to produce a final research report
- store_memory: Whether the latest observation contains useful factual information worth remembering for future reasoning.
  - Set true ONLY for: useful discoveries, factual tool outputs, insights relevant to the research question
  - Set false for: intermediate thoughts, malformed observations, tool errors, redundant information

RULES:
1. Output ONLY the JSON object. No markdown fences.
2. decision must be exactly "continue", "search_again", or "stop".
3. store_memory must be true or false (boolean, not string).
4. Do NOT invent fields. Use ONLY the three keys shown above.

Research Question:
{state.question}

Latest Thought:
{latest_thought}

Latest Action:
{latest_action}

Latest Observation:
{latest_observation}

Respond with a single JSON object (ONLY the JSON) now:"""