"""Test module for additional Agnassan model interfaces.

This module demonstrates how to use the additional model interfaces,
including Cohere, Grok, and Replicate.
"""

import os
import pytest
import asyncio
from pathlib import Path

from agnassan.config import ModelConfig
from agnassan.models import (
    create_model_interface,
    CohereInterface,
    GrokInterface,
    ReplicateInterface,
    LLMResponse
)

# Example configuration for testing with Cohere API
def test_cohere_api():
    """Demonstrate how to use Cohere API for inference."""
    config = ModelConfig(
        name="command",
        provider="cohere",
        api_key_env="COHERE_API_KEY",  # Set your API key in environment
        parameters={
            "model": "command",
            "temperature": 0.7,
            "max_tokens": 100
        },
        strengths=["general_knowledge", "creative"]
    )
    
    # Create model interface
    model = create_model_interface(config)
    
    # Test generation
    prompt = "Explain quantum computing in simple terms"
    response = asyncio.run(model.generate(prompt))
    
    assert isinstance(response, LLMResponse)
    assert response.text
    assert response.model_name == "command"
    assert response.tokens_used > 0

# Example configuration for testing with Grok API
def test_grok_api():
    """Demonstrate how to use Grok API for inference."""
    config = ModelConfig(
        name="grok-1",
        provider="grok",
        api_key_env="GROK_API_KEY",  # Set your API key in environment
        parameters={
            "model": "grok-1",
            "temperature": 0.7,
            "max_tokens": 100
        },
        strengths=["complex_reasoning", "coding"]
    )
    
    # Create model interface
    model = create_model_interface(config)
    
    # Test generation
    prompt = "Write a recursive function to calculate Fibonacci numbers"
    response = asyncio.run(model.generate(prompt))
    
    assert isinstance(response, LLMResponse)
    assert response.text
    assert response.model_name == "grok-1"
    assert response.tokens_used > 0

# Example configuration for testing with Replicate API
def test_replicate_api():
    """Demonstrate how to use Replicate API for inference."""
    config = ModelConfig(
        name="llama-2-70b",
        provider="replicate",
        api_key_env="REPLICATE_API_TOKEN",  # Set your API key in environment
        parameters={
            "model_path": "meta/llama-2-70b-chat",
            "temperature": 0.7,
            "max_tokens": 100,
            "system_prompt": "You are a helpful assistant."
        },
        strengths=["general_knowledge", "reasoning"]
    )
    
    # Create model interface
    model = create_model_interface(config)
    
    # Test generation
    prompt = "What are the ethical considerations of AI development?"
    response = asyncio.run(model.generate(prompt))
    
    assert isinstance(response, LLMResponse)
    assert response.text
    assert response.model_name == "llama-2-70b"
    assert response.tokens_used > 0

# Example of configuring multiple models in a YAML file
def test_yaml_config_example():
    """Demonstrate how to configure multiple models in a YAML file."""
    yaml_content = """
    models:
      - name: mistral-7b
        provider: local
        parameters:
          model_path: ./models/mistral-7b
          context_length: 4096
          temperature: 0.7
        strengths:
          - general_knowledge
          - coding
        cost_per_token: 0.0
      
      - name: gpt-4o
        provider: openai
        api_key_env: OPENAI_API_KEY
        parameters:
          model: gpt-4o
          temperature: 0.7
          max_tokens: 4096
        strengths:
          - complex_reasoning
          - creative
          - coding
        cost_per_token: 0.00001
      
      - name: command
        provider: cohere
        api_key_env: COHERE_API_KEY
        parameters:
          model: command
          temperature: 0.7
          max_tokens: 1000
        strengths:
          - general_knowledge
          - summarization
        cost_per_token: 0.000005
    
    default_model: mistral-7b
    log_dir: ./logs
    routing_strategy: rule_based
    """
    
    # In a real test, you would save this to a file and load it
    # from agnassan.config import AgnassanConfig
    # config = AgnassanConfig.from_yaml("config.yaml")
    
    print("Example YAML configuration for multiple models:")
    print(yaml_content)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])