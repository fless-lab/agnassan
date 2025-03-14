"""Extension for the reasoning module of Agnassan.

This module adds the missing apply_techniques method to the ReasoningEngine class.
"""

from typing import List, Dict, Any
from .reasoning import ReasoningEngine
from .models import LLMResponse, ModelInterface

# Extend the ReasoningEngine class with the missing method
async def apply_techniques(self, techniques: List[str], prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
    """Apply a sequence of reasoning techniques to enhance the model's response.
    
    Args:
        techniques: List of technique names to apply in sequence.
        prompt: The input prompt to process.
        model: The model interface to use for generation.
        **kwargs: Additional parameters to pass to the model.
        
    Returns:
        Enhanced response after applying the techniques.
        
    Raises:
        ValueError: If any of the specified techniques are unknown.
    """
    if not techniques:
        # If no techniques specified, just use the model directly
        return model.generate(prompt, **kwargs)
    
    # Apply the first technique
    current_prompt = prompt
    current_response = None
    
    for technique_name in techniques:
        if technique_name not in self.techniques:
            raise ValueError(f"Unknown reasoning technique: {technique_name}")
        
        # Apply the technique
        current_response = await self.techniques[technique_name].apply(current_prompt, model, **kwargs)
        
        # Update the prompt for the next technique if there is one
        current_prompt = current_response.text
    
    return current_response

# Monkey patch the ReasoningEngine class
ReasoningEngine.apply_techniques = apply_techniques