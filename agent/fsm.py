"""
Deterministic FSM core for the research agent.

Three pure functions:
  - transition(state, parsed, registry) → AgentState (THINK → ACT)
  - inject_observation(state, observation) → AgentState (ACT → OBSERVE → REFLECT)
  - reflect_transition(state, reflection) → (AgentState, bool) (REFLECT → DECIDE)
"""

from .state import AgentState
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# transition  (THINK → ACT)
# ---------------------------------------------------------------------------


def transition(state: AgentState, parsed_output: dict, registry: ToolRegistry) -> AgentState:
    """
    Centralized, deterministic state mutation for the THINK phase.

    Steps:
      1. Guard: if status != RUNNING, return immediately
      2. Increment iteration
      3. Check parse_error → ERROR
      4. Append thought
      5. Handle final_answer → FINISHED
      6. Handle tool → validate against registry, append action
      7. Set fsm_phase = ACT
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
        state.fsm_phase = "DECIDE"
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
        state.fsm_phase = "ACT"

    return state


# ---------------------------------------------------------------------------
# inject_observation  (ACT → OBSERVE → REFLECT)
# ---------------------------------------------------------------------------


def inject_observation(state: AgentState, observation: str) -> AgentState:
    """
    Append an observation to state.observations.
    Advances fsm_phase from ACT → REFLECT.
    Does NOT mutate iteration, status, or any other field.
    """
    state.observations.append(observation)
    state.fsm_phase = "REFLECT"
    return state


# ---------------------------------------------------------------------------
# reflect_transition  (REFLECT → DECIDE)
# ---------------------------------------------------------------------------


def reflect_transition(state: AgentState, reflection_output: dict) -> tuple:
    """
    Process the REFLECT phase output and decide next FSM move.

    Returns:
      (state, store_memory: bool)

    Reflection parse errors are handled as agent errors.
    """
    # Parse error guard
    if "parse_error" in reflection_output:
        state.status = "ERROR"
        state.error_reason = f"Reflection parse error: {reflection_output['parse_error']}"
        return state, False

    # Append critique to reflections log
    state.reflections.append(reflection_output["critique"])
    state.fsm_phase = "DECIDE"

    decision = reflection_output["decision"]
    store_memory = reflection_output["store_memory"]

    if decision == "stop":
        # Agent decides to stop — synthesize answer from accumulated knowledge
        state.status = "FINISHED"
        # final_answer will be set by a follow-up LLM call in the loop
        # if not already set
    elif decision in ("continue", "search_again"):
        state.fsm_phase = "THINK"
    else:
        state.status = "ERROR"
        state.error_reason = f"Unknown reflection decision: '{decision}'"
        return state, False

    return state, store_memory
