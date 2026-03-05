from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class AgentState:
    question: str
    thoughts: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 6
    status: str = "RUNNING"            # RUNNING | FINISHED | ERROR | MAX_ITER
    error_reason: Optional[str] = None