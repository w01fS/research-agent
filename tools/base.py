from abc import ABC, abstractmethod
from typing import Dict


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, input: Dict) -> str:
        """
        Executes the tool with structured input.
        Must return a string observation.
        """
        pass