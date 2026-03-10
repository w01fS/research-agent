"""
Deterministic FSM core for the research agent.

Two pure functions:
  - transition(state, parsed, registry) → AgentState (centralized state mutation)
  - inject_observation(state, observation) → AgentState (append observation only)
"""

from .state import AgentState
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------


def transition(state: AgentState, parsed_output: dict, registry: ToolRegistry) -> AgentState:
    """
    Centralized, deterministic state mutation.

    Steps:
      1. Guard: if status != RUNNING, return immediately
      2. Increment iteration
      3. Check max_iterations → MAX_ITER
      4. Check parse_error → ERROR
      5. Append thought
      6. Handle final_answer → FINISHED
      7. Handle tool → validate against registry, append action
    """
    # 1. Terminal-state guard
    if state.status != "RUNNING":
        return state

    # 2. Increment iteration
    state.iteration += 1

    # 3. Parse error check
    if "parse_error" in parsed_output:
        state.status = "ERROR"
        state.error_reason = parsed_output["parse_error"]
        return state

    # 4. Append thought
    state.thoughts.append(parsed_output["thought"])

    # 5. Final answer — check this BEFORE max iterations
    # (Allow the agent to finish on the last leg)
    if parsed_output["action_type"] == "final_answer":
        state.final_answer = parsed_output["final_answer"]
        state.status = "FINISHED"
        return state

    # 6. Max-iterations check for tool actions
    # (Stop if we can't take more actions)
    if state.iteration >= state.max_iterations:
        state.status = "MAX_ITER"
        state.error_reason = (
            f"Reached maximum iterations ({state.max_iterations}) without final answer."
        )
        return state

    # 7. Tool action & Loop detection
    if parsed_output["action_type"] == "tool":
        tool_name = parsed_output["tool_name"]
        tool_input = parsed_output["tool_input"]

        if registry.get(tool_name) is None:
            state.status = "ERROR"
            state.error_reason = f"Unknown tool: '{tool_name}'"
            return state

        # Loop Detection: check if we've already done this exact call
        current_action = {"tool_name": tool_name, "tool_input": tool_input}
        if current_action in state.actions:
            state.status = "ERROR"
            state.error_reason = f"Loop detected: identical tool call repeated: {tool_name}({tool_input})"
            return state

        state.actions.append(current_action)

    return state


# ---------------------------------------------------------------------------
# inject_observation
# ---------------------------------------------------------------------------


def inject_observation(state: AgentState, observation: str) -> AgentState:
    """
    Append an observation to state.observations.
    Does NOT mutate iteration, status, or any other field.
    """
    state.observations.append(observation)
    return state
