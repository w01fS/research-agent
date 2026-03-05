from .state import AgentState
from .prompt import build_prompt
from .parser import strict_json_parse
from .fsm import transition, inject_observation
from tools.registry import ToolRegistry


class AgentLoop:
    def __init__(self, llm_client, registry: ToolRegistry, max_iterations: int = 6):
        self.llm = llm_client
        self.registry = registry
        self.max_iterations = max_iterations

    def run(self, state: AgentState) -> AgentState:
        """
        FSM-driven agent loop.

        Returns the final AgentState. Caller inspects state.status and
        state.final_answer / state.error_reason.
        """
        state.max_iterations = self.max_iterations

        while state.status == "RUNNING":
            prompt = build_prompt(state, self.registry)
            raw_output = self.llm.generate(prompt)

            print(f"\n--- Iteration {state.iteration + 1} ---")
            print("RAW LLM OUTPUT:\n", raw_output)

            parsed = strict_json_parse(raw_output)
            state = transition(state, parsed, self.registry)

            if state.status != "RUNNING":
                break

            # Tool execution happens AFTER transition, only if still RUNNING
            if parsed.get("action_type") == "tool":
                observation = _execute_tool(
                    parsed["tool_name"], parsed["tool_input"], self.registry
                )
                state = inject_observation(state, observation)
                print(f"TOOL '{parsed['tool_name']}' → {observation}")

        return state


def _execute_tool(tool_name: str, tool_input: dict, registry: ToolRegistry) -> str:
    """
    Execute a tool and return its observation string.
    Tool existence is already validated by transition().
    """
    tool = registry.get(tool_name)
    try:
        return tool.run(tool_input)
    except Exception as e:
        return f"Tool execution error: {str(e)}"