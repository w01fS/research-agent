from pydantic import BaseModel
from typing import List, Optional

class AgentState(BaseModel):
    question: str
    iteration: int = 0
    max_iterations: int = 5
    
    thoughts: List[str] = []
    actions: List[str] = []
    observations: List[str] = []
    
    final_answer: Optional[str] = None
    terminated: bool = False