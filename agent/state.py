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
    def add_action(self, action: dict):
        self.actions.append(action)

    def add_thought(self, thought: str):
        self.thoughts.append(thought)

    def add_observation(self, observation: str):
        self.observations.append(observation)

    def increment_iteration(self):
        self.iteration += 1