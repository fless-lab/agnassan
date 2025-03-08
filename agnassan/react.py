"""ReAct module for Agnassan.

This module implements the ReAct (Reasoning + Acting) technique for enhanced problem-solving.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import re
import asyncio
import logging

from .models import LLMResponse, ModelInterface
from .reasoning import ReasoningTechnique


class Action:
    """Represents an action that can be taken during reasoning."""
    
    def __init__(self, name: str, function: Callable, description: str):
        self.name = name
        self.function = function
        self.description = description
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the action with the given arguments."""
        return await self.function(*args, **kwargs)


class ReActReasoning(ReasoningTechnique):
    """ReAct (Reasoning + Acting) technique.
    
    Combines reasoning with the ability to take actions, enabling more interactive
    and dynamic problem-solving capabilities.
    """
    
    def __init__(self, actions: Optional[List[Action]] = None, max_steps: int = 5):
        super().__init__("react")
        self.actions = actions or []
        self.max_steps = max_steps
        self.logger = logging.getLogger("agnassan.react")
    
    def register_action(self, action: Action) -> None:
        """Register a new action that can be used during reasoning."""
        self.actions.append(action)
    
    def _get_action_descriptions(self) -> str:
        """Get descriptions of all available actions."""
        descriptions = [f"{action.name}: {action.description}" for action in self.actions]
        return "\n".join(descriptions)
    
    def _parse_thought_action(self, text: str) -> tuple[str, Optional[str], List[str]]:
        """Parse the model's output into thought, action, and action arguments."""
        # Regular expressions to extract thought, action, and action arguments
        thought_pattern = r"Thought:(.*?)(?:Action:|$)"
        action_pattern = r"Action:(.*?)(?:Action Input:|$)"
        action_input_pattern = r"Action Input:(.*?)(?:$)"
        
        thought_match = re.search(thought_pattern, text, re.DOTALL)
        action_match = re.search(action_pattern, text, re.DOTALL)
        action_input_match = re.search(action_input_pattern, text, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else None
        action_input = action_input_match.group(1).strip() if action_input_match else ""
        action_args = [arg.strip() for arg in action_input.split(",") if arg.strip()]
        
        return thought, action, action_args
    
    async def _execute_action(self, action_name: str, action_args: List[str]) -> str:
        """Execute the specified action with the given arguments."""
        for action in self.actions:
            if action.name.lower() == action_name.lower():
                try:
                    result = await action.execute(*action_args)
                    return f"Action Result: {result}"
                except Exception as e:
                    return f"Error executing action: {str(e)}"
        
        return f"Unknown action: {action_name}"
    
    async def apply(self, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply ReAct reasoning to solve a problem through reasoning and acting."""
        # Prepare the initial prompt with action descriptions
        action_descriptions = self._get_action_descriptions()
        react_prompt = f"""{prompt}

You can use the following actions to help solve this problem:
{action_descriptions}

Use the following format:
Thought: Think about what to do next
Action: The action to take (one of {', '.join([action.name for action in self.actions])})
Action Input: The input to the action
Observation: The result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The final answer to the original question

Begin!
Thought:"""
        
        current_prompt = react_prompt
        steps = []
        total_tokens = 0
        final_answer = None
        
        # Execute the ReAct loop for a maximum number of steps
        for step in range(self.max_steps):
            # Generate the next thought and action
            response = await model.generate(current_prompt, **kwargs)
            total_tokens += response.tokens_used
            
            # Parse the response
            thought, action, action_args = self._parse_thought_action(response.text)
            
            # Check if we've reached a final answer
            if "Final Answer:" in response.text:
                final_answer_match = re.search(r"Final Answer:(.*?)$", response.text, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    steps.append({"thought": thought, "final_answer": final_answer})
                    break
            
            # If no action is specified, treat it as the final step
            if not action:
                final_answer = thought
                steps.append({"thought": thought})
                break
            
            # Execute the action and get the result
            observation = await self._execute_action(action, action_args)
            steps.append({"thought": thought, "action": action, "action_args": action_args, "observation": observation})
            
            # Update the prompt for the next iteration
            current_prompt = f"{current_prompt}\n{response.text}\n{observation}\nThought:"
        
        # Construct the final response
        if final_answer:
            result_text = f"After reasoning and taking actions, I've found the answer:\n\n{final_answer}"
        else:
            # If we reached the maximum steps without a final answer, use the last thought
            result_text = f"After {self.max_steps} steps of reasoning and actions, here's what I've found:\n\n{steps[-1]['thought']}"
        
        return LLMResponse(
            text=result_text,
            model_name=f"{model.name}_react",
            tokens_used=total_tokens,
            metadata={
                "reasoning_technique": "react",
                "steps": steps,
                "final_answer": final_answer
            }
        )


# Example actions that can be used with ReAct

async def search_web(query: str) -> str:
    """Simulated web search action."""
    # In a real implementation, this would connect to a search API
    return f"Search results for '{query}'"


async def calculate(expression: str) -> str:
    """Perform a calculation."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"


async def retrieve_document(doc_id: str) -> str:
    """Retrieve a document by ID."""
    # In a real implementation, this would retrieve from a database
    return f"Content of document {doc_id}"


# Create default actions
def create_default_actions() -> List[Action]:
    """Create a default set of actions for ReAct reasoning."""
    return [
        Action("search", search_web, "Search for information on the web"),
        Action("calculate", calculate, "Perform a mathematical calculation"),
        Action("retrieve", retrieve_document, "Retrieve a document by ID")
    ]