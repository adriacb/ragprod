"""The idea is to use a LLM to optimize the prompt for a given task."""

from .base import BasePromptOptimizer
from pydantic import BaseModel

class PromptOptimizer(BasePromptOptimizer, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def optimize(self, prompt: str, **kwargs) -> str:
        """Optimize the prompt for a given task.
        - remove unnecessary information
        - verifies ambiguity
        - uses an adapted terminology
        """
        pass