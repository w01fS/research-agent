from typing import List, Optional
from dataclasses import dataclass, field
from tools.registry import ToolRegistry

@dataclass
class AgentState:
    question: str
    thoughts: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 6
    terminated: bool = False

    # -----------------------------
    # Required by AgentLoop
    # -----------------------------
    def build_prompt(self, initial_prompt: str, registry: ToolRegistry) -> str:
        """
        Build the prompt for the LLM including:
        - initial question
        - short-term thoughts
        - previous observations
        - available tools
        """
        tool_list = ", ".join(registry.list_tools())
        short_term_memory = "\n".join(self.thoughts + self.observations)
        prompt = (
            f"Question: {initial_prompt}\n\n"
            f"Previous reasoning and observations:\n{short_term_memory}\n\n"
            f"Available tools: {tool_list}\n\n"
            "Respond with THOUGHT, ACTION, or FINAL in structured format."
        )
        return prompt

    def add_thought(self, thought: str):
        self.thoughts.append(thought)

    def add_observation(self, observation: str):
        self.observations.append(observation)