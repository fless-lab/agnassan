"""Reasoning module for Agnassan.

This module implements advanced reasoning techniques to enhance LLM responses.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import asyncio

from .models import LLMResponse, ModelInterface


class ReasoningTechnique:
    """Base class for reasoning techniques."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def apply(self, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply the reasoning technique to enhance the model's response."""
        raise NotImplementedError("Subclasses must implement this method")


class ChainOfThought(ReasoningTechnique):
    """Chain of Thought reasoning technique.
    
    Breaks down complex problems into sequential steps, making the reasoning process explicit.
    """
    
    def __init__(self):
        super().__init__("chain_of_thought")
    
    async def apply(self, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply Chain of Thought reasoning."""
        # Enhance the prompt to encourage step-by-step thinking
        enhanced_prompt = f"{prompt}\n\nLet's solve this step-by-step:"
        
        # Generate the response with the enhanced prompt
        response = await model.generate(enhanced_prompt, **kwargs)
        
        return LLMResponse(
            text=response.text,
            model_name=f"{model.name}_chain_of_thought",
            tokens_used=response.tokens_used,
            metadata={**response.metadata, "reasoning_technique": "chain_of_thought"}
        )


class TreeOfThought(ReasoningTechnique):
    """Tree of Thought reasoning technique.
    
    Explores multiple reasoning paths simultaneously and selects the most promising one.
    """
    
    def __init__(self, num_branches: int = 3, evaluation_prompt: str = "Which of the above solutions is most accurate and complete?"):
        super().__init__("tree_of_thought")
        self.num_branches = num_branches
        self.evaluation_prompt = evaluation_prompt
    
    async def apply(self, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply Tree of Thought reasoning."""
        # Generate multiple reasoning paths
        branch_prompts = [
            f"{prompt}\n\nApproach {i+1}: Let's think about this from the perspective of {perspective}"
            for i, perspective in enumerate(["physics", "history", "practical applications"][:self.num_branches])
        ]
        
        # Generate responses for each branch
        branch_responses = await asyncio.gather(*[
            model.generate(branch_prompt, **kwargs)
            for branch_prompt in branch_prompts
        ])
        
        # Combine the branches
        combined_text = "\n\n".join([f"Approach {i+1}:\n{response.text}" for i, response in enumerate(branch_responses)])
        
        # Evaluate the branches to select the best one
        evaluation_prompt = f"{combined_text}\n\n{self.evaluation_prompt}"
        evaluation = await model.generate(evaluation_prompt, **kwargs)
        
        # Extract the final answer based on the evaluation
        final_text = f"I explored multiple approaches to this question:\n\n{combined_text}\n\nBased on these approaches, here's my conclusion:\n{evaluation.text}"
        
        total_tokens = sum(response.tokens_used for response in branch_responses) + evaluation.tokens_used
        
        return LLMResponse(
            text=final_text,
            model_name=f"{model.name}_tree_of_thought",
            tokens_used=total_tokens,
            metadata={"reasoning_technique": "tree_of_thought", "num_branches": self.num_branches}
        )


class IterativeRefinement(ReasoningTechnique):
    """Iterative Refinement reasoning technique.
    
    Progressively refines the response through multiple iterations.
    """
    
    def __init__(self, num_iterations: int = 3, refinement_prompt: str = "Please improve the above solution by fixing any errors and adding more details."):
        super().__init__("iterative_refinement")
        self.num_iterations = num_iterations
        self.refinement_prompt = refinement_prompt
    
    async def apply(self, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply Iterative Refinement reasoning."""
        current_prompt = prompt
        iterations = []
        total_tokens = 0
        
        # Perform multiple iterations of refinement
        for i in range(self.num_iterations):
            response = await model.generate(current_prompt, **kwargs)
            iterations.append(response.text)
            total_tokens += response.tokens_used
            
            # Update the prompt for the next iteration
            current_prompt = f"{prompt}\n\nCurrent solution:\n{response.text}\n\n{self.refinement_prompt}"
        
        # Combine the iterations to show the progression
        final_text = f"Through {self.num_iterations} iterations of refinement, I arrived at this solution:\n\n{iterations[-1]}"
        
        return LLMResponse(
            text=final_text,
            model_name=f"{model.name}_iterative_refinement",
            tokens_used=total_tokens,
            metadata={"reasoning_technique": "iterative_refinement", "iterations": iterations}
        )


class ParallelReasoning(ReasoningTechnique):
    """Parallel Reasoning technique.
    
    Distributes different aspects of a problem to different models and combines their responses.
    """
    
    def __init__(self, models: List[ModelInterface], combination_strategy: str = "synthesis"):
        super().__init__("parallel_reasoning")
        self.models = models
        self.combination_strategy = combination_strategy
    
    async def apply(self, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply Parallel Reasoning."""
        # Generate responses from multiple models in parallel
        responses = await asyncio.gather(*[
            model.generate(prompt, **kwargs) for model in self.models
        ])
        
        # Combine the responses based on the selected strategy
        if self.combination_strategy == "synthesis":
            # Use the main model to synthesize the responses
            synthesis_prompt = f"{prompt}\n\nI have received the following responses:\n\n"
            for i, response in enumerate(responses):
                synthesis_prompt += f"Response {i+1} (from {response.model_name}):\n{response.text}\n\n"
            synthesis_prompt += "Please synthesize these responses into a comprehensive answer."
            
            synthesis = await model.generate(synthesis_prompt, **kwargs)
            final_text = synthesis.text
            total_tokens = sum(response.tokens_used for response in responses) + synthesis.tokens_used
        else:  # Simple concatenation
            final_text = "\n\n".join([f"From {response.model_name}:\n{response.text}" for response in responses])
            total_tokens = sum(response.tokens_used for response in responses)
        
        return LLMResponse(
            text=final_text,
            model_name="parallel_reasoning",
            tokens_used=total_tokens,
            metadata={"reasoning_technique": "parallel_reasoning", "models_used": [r.model_name for r in responses]}
        )


class MetaCritique(ReasoningTechnique):
    """Meta-Critique reasoning technique.
    
    Critically evaluates and improves the initial response.
    """
    
    def __init__(self, critique_prompt: str = "Please critically evaluate the above response. Identify any errors, biases, or areas for improvement."):
        super().__init__("meta_critique")
        self.critique_prompt = critique_prompt
    
    async def apply(self, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply Meta-Critique reasoning."""
        # Generate the initial response
        initial_response = await model.generate(prompt, **kwargs)
        
        # Generate a critique of the response
        critique_prompt = f"Original question: {prompt}\n\nResponse: {initial_response.text}\n\n{self.critique_prompt}"
        critique = await model.generate(critique_prompt, **kwargs)
        
        # Generate an improved response based on the critique
        improvement_prompt = f"Original question: {prompt}\n\nInitial response: {initial_response.text}\n\nCritique: {critique.text}\n\nPlease provide an improved response that addresses the critique."
        improved_response = await model.generate(improvement_prompt, **kwargs)
        
        final_text = f"After careful consideration and self-critique, here's my response:\n\n{improved_response.text}"
        total_tokens = initial_response.tokens_used + critique.tokens_used + improved_response.tokens_used
        
        return LLMResponse(
            text=final_text,
            model_name=f"{model.name}_meta_critique",
            tokens_used=total_tokens,
            metadata={
                "reasoning_technique": "meta_critique",
                "initial_response": initial_response.text,
                "critique": critique.text
            }
        )


class ReasoningEngine:
    """Engine for applying reasoning techniques to enhance LLM responses."""
    
    def __init__(self):
        self.techniques = {
            "chain_of_thought": ChainOfThought(),
            "tree_of_thought": TreeOfThought(),
            "iterative_refinement": IterativeRefinement(),
            "meta_critique": MetaCritique()
        }
        
        # Try to import and register ReAct reasoning if available
        try:
            from .react import ReActReasoning, create_default_actions
            react_technique = ReActReasoning(actions=create_default_actions())
            self.techniques["react"] = react_technique
        except ImportError:
            pass
            
        # Register advanced reasoning techniques
        try:
            from .parallel_thought import register_advanced_techniques
            register_advanced_techniques(self)
        except ImportError:
            pass
    
    def register_technique(self, technique: ReasoningTechnique) -> None:
        """Register a new reasoning technique."""
        self.techniques[technique.name] = technique
    
    async def apply_technique(self, technique_name: str, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply a specific reasoning technique."""
        if technique_name not in self.techniques:
            raise ValueError(f"Unknown reasoning technique: {technique_name}")
        
        return await self.techniques[technique_name].apply(prompt, model, **kwargs)
    
    async def apply_best_technique(self, prompt: str, model: ModelInterface, **kwargs) -> LLMResponse:
        """Apply the most appropriate reasoning technique based on the prompt."""
        # Simple heuristic-based selection for now
        # In a more advanced implementation, this could use ML to select the best technique
        if "step by step" in prompt.lower() or "explain" in prompt.lower():
            return await self.techniques["chain_of_thought"].apply(prompt, model, **kwargs)
        elif "different perspectives" in prompt.lower() or "multiple approaches" in prompt.lower():
            return await self.techniques["tree_of_thought"].apply(prompt, model, **kwargs)
        elif "improve" in prompt.lower() or "refine" in prompt.lower():
            return await self.techniques["iterative_refinement"].apply(prompt, model, **kwargs)
        elif "critique" in prompt.lower() or "evaluate" in prompt.lower():
            return await self.techniques["meta_critique"].apply(prompt, model, **kwargs)
        elif "action" in prompt.lower() or "do something" in prompt.lower() or "search" in prompt.lower():
            # Use ReAct if available and the prompt suggests actions
            if "react" in self.techniques:
                return await self.techniques["react"].apply(prompt, model, **kwargs)
            else:
                return await self.techniques["chain_of_thought"].apply(prompt, model, **kwargs)
        else:
            # Default to Chain of Thought as it's generally useful
            return await self.techniques["chain_of_thought"].apply(prompt, model, **kwargs)