"""Test module for Agnassan model interfaces.

This module demonstrates how to use different model interfaces,
including Hugging Face API and local models.
"""

import os
import pytest
import asyncio
from pathlib import Path

from agnassan.config import ModelConfig
from agnassan.models import (
    create_model_interface,
    OpenAIInterface,
    AnthropicInterface,
    LocalModelInterface,
    LLMResponse
)

# Example configuration for testing with Hugging Face API
def test_huggingface_api():
    """Demonstrate how to use Hugging Face API for inference."""
    config = ModelConfig(
        name="gpt2",
        provider="huggingface",
        api_key_env="HUGGINGFACE_API_KEY",  # Set your API key in environment
        parameters={
            "model": "gpt2",
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    
    # Create model interface
    model = create_model_interface(config)
    
    # Test generation
    prompt = "Once upon a time"
    response = asyncio.run(model.generate(prompt))
    
    assert isinstance(response, LLMResponse)
    assert response.text
    assert response.model_name == "gpt2"
    assert response.tokens_used > 0

# Example configuration for testing with local models
def test_local_model():
    """Demonstrate how to use downloaded local models."""
    # Assuming models are downloaded to ./models directory
    model_path = Path("./models/gpt2")
    
    config = ModelConfig(
        name="local-gpt2",
        provider="local",
        parameters={
            "model_path": str(model_path),
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    
    # Create model interface
    model = create_model_interface(config)
    
    # Test generation
    prompt = "The future of AI is"
    response = asyncio.run(model.generate(prompt))
    
    assert isinstance(response, LLMResponse)
    assert response.text
    assert response.model_name == "local-gpt2"
    assert response.tokens_used > 0

# Helper function to download models
def download_model(model_name: str, save_path: Path):
    """Download a model from Hugging Face to local storage.
    
    Args:
        model_name: Name of the model on Hugging Face Hub
        save_path: Local path to save the model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Download model and tokenizer
    print(f"Downloading {model_name} to {save_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save to local directory
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Model {model_name} downloaded successfully!")

# Example usage of model download
def test_model_download():
    """Demonstrate how to download and use a model locally."""
    model_name = "gpt2"  # or any other model from Hugging Face
    save_path = Path("./models/gpt2")
    
    if not save_path.exists():
        download_model(model_name, save_path)
    
    # Now use the local model
    config = ModelConfig(
        name="downloaded-gpt2",
        provider="local",
        parameters={
            "model_path": str(save_path),
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    
    model = create_model_interface(config)
    prompt = "Testing the downloaded model:"
    response = asyncio.run(model.generate(prompt))
    
    assert isinstance(response, LLMResponse)
    assert response.text
    assert response.model_name == "downloaded-gpt2"
    assert response.tokens_used > 0

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])