"""Advanced Reasoning Techniques Demo for Agnassan.

This script demonstrates how to use the advanced reasoning techniques
including Chain of Thought, Tree of Thought, Parallel Thought Chains,
and Iterative Loops with multiple open-source models.
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import agnassan
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agnassan.config import AgnassanConfig, create_default_config
from agnassan.routing import Router
from agnassan.reasoning import ReasoningEngine
from agnassan.parallel_thought import ParallelThoughtChains, IterativeLoops

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample queries to demonstrate different reasoning techniques
SAMPLE_QUERIES = {
    "chain_of_thought": "Explain how photosynthesis works step by step.",
    "tree_of_thought": "Consider different approaches to solving climate change.",
    "meta_critique": "Evaluate the strengths and weaknesses of democracy as a political system.",
    "parallel_thought_chains": "Provide a comprehensive analysis of artificial intelligence's impact on society, considering multiple perspectives.",
    "iterative_loops": "Progressively refine an explanation of quantum computing for a general audience.",
    "react": "Find information about the population of Tokyo and calculate what percentage of Japan's population that represents."
}


async def demonstrate_reasoning_technique(router, technique_name, query):
    """Demonstrate a specific reasoning technique with a sample query."""
    print(f"\n{'=' * 80}")
    print(f"Demonstrating {technique_name.upper()} reasoning technique")
    print(f"Query: {query}")
    print(f"{'=' * 80}\n")
    
    # Force the router to use the specified technique
    router.reasoning_selector.select_technique = lambda _: technique_name
    
    # Process the query
    response = await router.route_query(query, strategy="auto")
    
    print(f"\nResponse:\n{response.text}\n")
    print(f"Model used: {response.model_name}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"{'=' * 80}\n")
    
    return response


async def main():
    """Run the demonstration of advanced reasoning techniques."""
    print("\nAGNASSAN ADVANCED REASONING TECHNIQUES DEMONSTRATION\n")
    
    # Load the configuration with multiple open-source models
    config = create_default_config()
    
    # Create the router
    router = Router(config)
    
    # Ensure the advanced reasoning techniques are registered
    reasoning_engine = router.reasoning_engine
    
    # Verify that all techniques are registered
    print("Available reasoning techniques:")
    for technique_name in reasoning_engine.techniques:
        print(f"- {technique_name}")
    
    # Demonstrate each reasoning technique
    for technique_name, query in SAMPLE_QUERIES.items():
        if technique_name in reasoning_engine.techniques:
            await demonstrate_reasoning_technique(router, technique_name, query)
        else:
            print(f"Technique '{technique_name}' not available.")
    
    # Clean up
    await router.close()


if __name__ == "__main__":
    # Create the examples directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Run the demonstration
    asyncio.run(main())