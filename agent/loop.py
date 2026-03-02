from typing import Optional

from .parser import OutputParser
from .state import AgentState
from tools.registry import ToolRegistry


class AgentLoop:
    def __init__(self, llm_client, registry: ToolRegistry, max_iterations: int = 6):
        self.llm = llm_client
        self.registry = registry
        self.parser = OutputParser()
        self.max_iterations = max_iterations

    def run(self, initial_prompt: str, state: AgentState) -> str:
        """
        Executes the deterministic ReAct loop.
        """

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            prompt = state.build_prompt(initial_prompt, self.registry)
            raw_output = self.llm.generate(prompt)

            print("\nRAW LLM OUTPUT:\n", raw_output)

            parsed = self.parser.parse(raw_output)

            # Always log thought (parser guarantees string)
            state.add_thought(parsed.thought)

            if parsed.error:
                error_msg = f"Parse error: {parsed.error}"
                print(error_msg)
                state.add_observation(error_msg)
                continue

            # FINAL branch
            if parsed.final:
                if state.iteration == 0:
                    print("Rejecting premature FINAL.")
                    continue
                print(parsed.final)
                return parsed.final

            # ACTION branch
            if parsed.action:
                tool_name = parsed.action["tool"]
                tool_input = parsed.action["input"]

                tool = self.registry.get(tool_name)

                if not tool:
                    error_msg = f"Unknown tool: {tool_name}"
                    print(error_msg)
                    state.add_observation(error_msg)
                    continue

                try:
                    result = tool.run(tool_input)
                except Exception as e:
                    error_msg = f"Tool error: {str(e)}"
                    print(error_msg)
                    state.add_observation(error_msg)
                    continue

                print(f"Tool '{tool_name}' executed.")
                print("OBSERVATION:", result)

                state.add_observation(result)
                continue

        # If loop exits without FINAL
        fallback = "Max iterations reached without FINAL."
        print("\nFINAL ANSWER:")
        print(fallback)
        return fallback