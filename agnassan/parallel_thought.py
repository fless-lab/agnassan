"""Parallel Thought Chains module for Agnassan.

This module implements the Parallel Thought Chains technique for enhanced problem-solving
by running multiple reasoning techniques simultaneously and synthesizing their results.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
import logging

from .models import LLMResponse, ModelInterface
from .reasoning import ReasoningTechnique, ReasoningEngine


class ParallelThoughtChains(ReasoningTechnique):
    """Parallel Thought Chains reasoning technique.
    
    Applies multiple reasoning techniques in parallel and synthesizes their results,
    allowing for more comprehensive problem-solving by leveraging different cognitive approaches.
    """
    
    def __init__(self, techniques: List[str] = None, max_parallel: int = 3):
        super().__init__("parallel_thought_chains")
        self.techniques = techniques or ["chain_of_thought", "tree_of_thought", "meta_critique"]
        self.max_parallel = max_parallel
        self.logger = logging.getLogger("agnassan.parallel_thought")
    
    async def apply(self, prompt: str, model: ModelInterface, reasoning_engine: ReasoningEngine, **kwargs) -> LLMResponse:
        """Apply multiple reasoning techniques in parallel and synthesize the results."""
        # Limit the number of parallel techniques
        techniques_to_use = self.techniques[:self.max_parallel]
        self.logger.info(f"Applying parallel thought chains with techniques: {techniques_to_use}")
        
        # Apply each technique in parallel
        tasks = []
        for technique_name in techniques_to_use:
            if technique_name in reasoning_engine.techniques:
                tasks.append(reasoning_engine.apply_technique(technique_name, prompt, model, **kwargs))
            else:
                self.logger.warning(f"Technique {technique_name} not found in reasoning engine")
        
        if not tasks:
            self.logger.warning("No valid techniques found for parallel thought chains")
            # Fall back to chain of thought
            return await reasoning_engine.apply_technique("chain_of_thought", prompt, model, **kwargs)
        
        # Execute all techniques in parallel
        responses = await asyncio.gather(*tasks)
        
        # Create a synthesis prompt that includes all the responses
        synthesis_prompt = f"{prompt}\n\nI have analyzed this problem using different reasoning approaches:\n\n"
        
        for i, (response, technique) in enumerate(zip(responses, techniques_to_use)):
            synthesis_prompt += f"Approach {i+1} ({technique}):\n{response.text}\n\n"
        
        synthesis_prompt += "Based on these different approaches, please provide a comprehensive synthesis that combines the strengths of each approach and addresses any contradictions."
        
        # Generate the final synthesized response
        synthesis_response = await model.generate(synthesis_prompt, **kwargs)
        
        # Calculate total tokens used
        total_tokens = sum(response.tokens_used for response in responses) + synthesis_response.tokens_used
        
        return LLMResponse(
            text=synthesis_response.text,
            model_name=f"{model.name}_parallel_thought_chains",
            tokens_used=total_tokens,
            metadata={
                "reasoning_technique": "parallel_thought_chains",
                "techniques_used": techniques_to_use,
                "individual_responses": [response.to_dict() for response in responses]
            }
        )


class IterativeLoops(ReasoningTechnique):
    """Iterative Loops reasoning technique.
    
    Applies a sequence of different reasoning techniques iteratively, with each technique
    building upon the results of the previous one, creating a loop of increasingly refined answers.
    """
    
    def __init__(self, technique_sequence: List[str] = None, num_iterations: int = 2):
        super().__init__("iterative_loops")
        self.technique_sequence = technique_sequence or ["chain_of_thought", "meta_critique", "iterative_refinement"]
        self.num_iterations = num_iterations
        self.logger = logging.getLogger("agnassan.iterative_loops")
    
    async def apply(self, prompt: str, model: ModelInterface, reasoning_engine: ReasoningEngine, **kwargs) -> LLMResponse:
        """Apply a sequence of reasoning techniques iteratively."""
        current_prompt = prompt
        iterations = []
        total_tokens = 0
        
        for iteration in range(self.num_iterations):
            self.logger.info(f"Starting iteration {iteration+1} of iterative loops")
            
            # Apply each technique in the sequence
            for technique_name in self.technique_sequence:
                if technique_name not in reasoning_engine.techniques:
                    self.logger.warning(f"Technique {technique_name} not found in reasoning engine")
                    continue
                
                response = await reasoning_engine.apply_technique(technique_name, current_prompt, model, **kwargs)
                total_tokens += response.tokens_used
                
                # Update the prompt for the next technique with the results of this one
                current_prompt = f"{prompt}\n\nPrevious analysis:\n{response.text}\n\nPlease build upon this analysis to provide an even more comprehensive answer."
                
                iterations.append({
                    "iteration": iteration + 1,
                    "technique": technique_name,
                    "response": response.text
                })
        
        # The final response is the result of the last technique applied
        final_text = f"After applying multiple reasoning techniques iteratively, here is my comprehensive answer:\n\n{iterations[-1]['response']}"
        
        return LLMResponse(
            text=final_text,
            model_name=f"{model.name}_iterative_loops",
            tokens_used=total_tokens,
            metadata={
                "reasoning_technique": "iterative_loops",
                "technique_sequence": self.technique_sequence,
                "iterations": iterations
            }
        )


def register_advanced_techniques(reasoning_engine: ReasoningEngine) -> None:
    """Register advanced reasoning techniques with the reasoning engine."""
    # Register Parallel Thought Chains
    parallel_thought_chains = ParallelThoughtChains()
    reasoning_engine.register_technique(parallel_thought_chains)
    
    # Register Iterative Loops
    iterative_loops = IterativeLoops()
    reasoning_engine.register_technique(iterative_loops)