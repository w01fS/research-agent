"""
Agent loop: orchestrates the full FSM cycle.

THINK → ACT → OBSERVE → REFLECT → DECIDE → THINK | STOP

Handles:
  - Memory retrieval (FAISS) before THINK
  - Tool execution in ACT
  - Reflection via LLM in REFLECT
  - Memory write policy enforcement after DECIDE
"""

from .state import AgentState
from .prompt import build_think_prompt, build_reflect_prompt
from .parser import strict_json_parse
from .reflection import strict_reflection_parse
from .fsm import transition, inject_observation, reflect_transition
from tools.registry import ToolRegistry
from memory.embedding import EmbeddingModel
from memory.vector_store import VectorMemory


class AgentLoop:
    def __init__(
        self,
        llm_client,
        registry: ToolRegistry,
        embedder: EmbeddingModel,
        memory: VectorMemory,
        max_iterations: int = 6,
    ):
        self.llm = llm_client
        self.registry = registry
        self.embedder = embedder
        self.memory = memory
        self.max_iterations = max_iterations

    def run(self, state: AgentState) -> AgentState:
        """
        FSM-driven agent loop with reflection and memory.

        Returns the final AgentState. Caller inspects state.status and
        state.final_answer / state.error_reason.
        """
        state.max_iterations = self.max_iterations

        while state.status == "RUNNING":

            # ---------------------------------------------------------------
            # 1. MEMORY RETRIEVAL
            # ---------------------------------------------------------------
            query_text = state.question
            if state.thoughts:
                query_text += " " + state.thoughts[-1]
            query_embedding = self.embedder.embed(query_text)
            memories = self.memory.search(query_embedding, k=3)

            if memories:
                print(f"\n  [MEMORY] Retrieved {len(memories)} relevant memory(ies):")
                for i, mem in enumerate(memories, 1):
                    print(f"           [{i}] {mem[:120]}{'...' if len(mem) > 120 else ''}")

            # ---------------------------------------------------------------
            # 2. THINK  (LLM call → parse → transition)
            # ---------------------------------------------------------------
            prompt = build_think_prompt(state, self.registry, memories)
            raw_output = self.llm.generate(prompt)

            print(f"\n--- Iteration {state.iteration + 1} ---")
            print(f"  [THINK] LLM output:\n{raw_output}")

            parsed = strict_json_parse(raw_output)
            state = transition(state, parsed, self.registry)

            # If status changed (FINISHED, ERROR, MAX_ITER), exit
            if state.status != "RUNNING":
                break

            # ---------------------------------------------------------------
            # 3. ACT  (tool execution)
            # ---------------------------------------------------------------
            if parsed.get("action_type") == "tool":
                observation = _execute_tool(
                    parsed["tool_name"], parsed["tool_input"], self.registry
                )

                # Observation-level duplicate detection
                if observation in state.observations:
                    observation = (
                        f"[DUPLICATE] This tool returned the same result as a previous call. "
                        f"Original result: {observation[:200]}... "
                        f"Try a fundamentally different approach or produce your final answer."
                    )
                    print(f"  [ACT]     Tool '{parsed['tool_name']}' returned duplicate observation")
                else:
                    print(f"  [ACT]     Tool '{parsed['tool_name']}' executed")

                state = inject_observation(state, observation)
                print(f"  [OBSERVE] Result: {observation}")

            # ---------------------------------------------------------------
            # 4. REFLECT  (LLM call → parse reflection)
            # ---------------------------------------------------------------
            reflect_prompt = build_reflect_prompt(state)
            reflect_raw = self.llm.generate(reflect_prompt)

            print(f"  [REFLECT] LLM output:\n{reflect_raw}")

            reflection = strict_reflection_parse(reflect_raw)
            state, store_memory = reflect_transition(state, reflection)

            if "parse_error" not in reflection:
                print(f"  [DECIDE]  decision={reflection['decision']}, "
                      f"store_memory={reflection['store_memory']}")
                print(f"  [REFLECT] critique: {reflection['critique']}")

            # If reflection decided to stop, we need a final answer
            if state.status == "FINISHED" and state.final_answer is None:
                state = self._synthesize_final_answer(state, memories)
                break

            if state.status != "RUNNING":
                break

            # ---------------------------------------------------------------
            # 5. MEMORY WRITE  (deterministic, based on reflection decision)
            # ---------------------------------------------------------------
            if store_memory and state.observations:
                last_obs = state.observations[-1]
                # Never store error observations
                if not str(last_obs).startswith("Tool execution error:"):
                    embedding = self.embedder.embed(last_obs)
                    self.memory.add(embedding, last_obs)
                    print(f"  [MEMORY]  Stored observation to long-term memory")
                else:
                    print(f"  [MEMORY]  Skipped storing error observation")

        return state

    def _synthesize_final_answer(self, state: AgentState, memories: list) -> AgentState:
        """
        When the REFLECT phase decides to stop but no final_answer exists yet,
        run one more THINK call to synthesize the research report.
        """
        print("\n  [SYNTHESIZE] Generating final research report...")
        prompt = build_think_prompt(state, self.registry, memories)
        raw_output = self.llm.generate(prompt)
        print(f"  [SYNTHESIZE] LLM output:\n{raw_output}")

        parsed = strict_json_parse(raw_output)
        if parsed.get("action_type") == "final_answer":
            state.final_answer = parsed["final_answer"]
            state.status = "FINISHED"
        else:
            # If LLM still doesn't give a final answer, force-finish
            # with accumulated reflections
            state.final_answer = (
                "Research completed. Key findings from reflections:\n"
                + "\n".join(f"- {r}" for r in state.reflections)
            )
            state.status = "FINISHED"
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