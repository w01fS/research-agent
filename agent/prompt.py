from agent.state import AgentState
from tools.registry import ToolRegistry


def build_prompt(state: AgentState, registry: ToolRegistry) -> str:
    """
    Build the LLM prompt enforcing strict JSON output contract.
    """
    # --- Tool descriptions ---
    tools_desc = ""
    for name, tool in registry.list_tools().items():
        tools_desc += f'  - "{name}": {tool.description}\n'

    # --- Conversation history ---
    import json
    history = ""
    for i in range(len(state.thoughts)):
        history += f"PREVIOUS THOUGHT: {state.thoughts[i]}\n"
        if i < len(state.actions):
            # Show the action as the JSON object it matched
            action_json = json.dumps(state.actions[i])
            history += f"PREVIOUS ACTION: {action_json}\n"
        if i < len(state.observations):
            history += f"SYSTEM RESULT: {state.observations[i]}\n"
        history += "\n"

    return f"""You are a research agent. Answer questions by using tools step-by-step.

YOUR OUTPUT MUST BE A SINGLE RAW JSON OBJECT. No markdown, no explanation, no extra text.

Available tools:
{tools_desc}

Choose ONE of these two JSON formats:

Format A — to call a tool:
{{"thought": "<your reasoning>", "action_type": "tool", "tool_name": "<tool>", "tool_input": {{"<key>": "<value>"}}}}

Example (search_local_knowledge):
{{"thought": "I need to search for semiconductor info.", "action_type": "tool", "tool_name": "search_local_knowledge", "tool_input": {{"query": "semiconductor"}}}}

Example (calculator):
{{"thought": "I need to calculate 2 + 2.", "action_type": "tool", "tool_name": "calculator", "tool_input": {{"expression": "2+2"}}}}

Format B — to give a final answer:
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

STOPPING CRITERIA:
- If the CONVERSATION HISTORY below already contains a 'SYSTEM RESULT' with the information needed to answer the question, you MUST use Format B immediately.
- DO NOT repeat the same tool call with the same input if you already have its result.

Question:
{state.question}

CONVERSATION HISTORY:
{history if history else "(none yet)"}

Iteration: {state.iteration}/{state.max_iterations}

Respond with a single JSON object (ONLY the JSON) now:"""