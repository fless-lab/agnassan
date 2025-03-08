"""Model interface module for Agnassan.

This module provides interfaces for different language model providers.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

from .config import ModelConfig


class LLMResponse:
    """Represents a response from a language model.
    
    Attributes:
        text: The generated text response.
        model_name: Name of the model that generated the response.
        tokens_used: Number of tokens used for this response.
        metadata: Additional metadata about the response.
    """
    
    def __init__(
        self,
        text: str,
        model_name: str,
        tokens_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.model_name = model_name
        self.tokens_used = tokens_used
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary."""
        return {
            "text": self.text,
            "model_name": self.model_name,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata
        }


class ModelInterface(ABC):
    """Abstract base class for language model interfaces."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the language model."""
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        pass


class OpenAIInterface(ModelInterface):
    """Interface for OpenAI models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url
            )
        except ImportError:
            raise ImportError("OpenAI package is required for OpenAI models. Install with 'pip install openai'.")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using an OpenAI model."""
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        # Get model ID from config and look it up in the registry
        model_id = params.get("model_id", self.config.name)
        model_registry = params.get("model_registry", {})
        
        # Look up the model name in the registry or use the model_id as the name
        model_name = model_registry.get(model_id, model_id)
        
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 1000),
        )
        
        return LLMResponse(
            text=response.choices[0].message.content,
            model_name=self.name,
            tokens_used=response.usage.total_tokens,
            metadata={"response": response}
        )
    
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.parameters.get("model", "gpt-3.5-turbo"))
            return len(encoding.encode(text))
        except ImportError:
            raise ImportError("Tiktoken package is required for token counting. Install with 'pip install tiktoken'.")


class AnthropicInterface(ModelInterface):
    """Interface for Anthropic models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=config.api_key
            )
        except ImportError:
            raise ImportError("Anthropic package is required for Anthropic models. Install with 'pip install anthropic'.")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using an Anthropic model."""
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        # Get model ID from config and look it up in the registry
        model_id = params.get("model_id", self.config.name)
        model_registry = params.get("model_registry", {})
        
        # Look up the model name in the registry or use the model_id as the name
        model_name = model_registry.get(model_id, model_id)
        
        response = await self.client.messages.create(
            model=model_name,
            max_tokens=params.get("max_tokens", 1000),
            temperature=params.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return LLMResponse(
            text=response.content[0].text,
            model_name=self.name,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            metadata={"response": response}
        )
    
    async def count_tokens(self, text: str) -> int:
        """Approximate token count for Anthropic models."""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4


class LocalModelInterface(ModelInterface):
    """Interface for local language models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the local model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import logging
            
            # Get model ID from config and look it up in the registry
            model_id = self.config.parameters.get("model_id")
            model_registry = self.config.parameters.get("model_registry", {})
            
            if not model_id:
                raise ValueError(f"Model ID not specified for local model {self.name}")
            
            # Look up the model path in the registry or use the model_id as a path
            model_path = model_registry.get(model_id, model_id)
            logging.getLogger("agnassan.models").info(f"Loading local model from: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except ImportError:
            raise ImportError("Transformers and PyTorch are required for local models. Install with 'pip install transformers torch'.")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using a local model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError(f"Model {self.name} is not loaded properly")
        
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = len(inputs.input_ids[0])
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
                do_sample=True
            )
        
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        response_text = output_text[len(prompt):].strip()
        output_tokens = len(outputs[0]) - input_tokens
        
        return LLMResponse(
            text=response_text,
            model_name=self.name,
            tokens_used=input_tokens + output_tokens,
            metadata={}
        )
    
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using the model's tokenizer."""
        if not self.tokenizer:
            raise RuntimeError(f"Tokenizer for model {self.name} is not loaded properly")
        
        return len(self.tokenizer.encode(text))


class HuggingFaceInterface(ModelInterface):
    """Interface for Hugging Face models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=config.api_key)
        except ImportError:
            raise ImportError("Hugging Face Hub package is required for Hugging Face models. Install with 'pip install huggingface_hub'")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using a Hugging Face model."""
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        # Get model ID from config and look it up in the registry
        model_id = params.get("model_id", self.config.name)
        model_registry = params.get("model_registry", {})
        
        # Look up the model name in the registry or use the model_id as the name
        model_name = model_registry.get(model_id, model_id)
        
        # Use asyncio to run the synchronous API call in a thread pool
        import asyncio
        import functools
        
        # Define a synchronous function to call the Hugging Face API
        def _generate():
            response = self.client.text_generation(
                prompt,
                model=model_name,
                max_new_tokens=params.get("max_tokens", 100),
                temperature=params.get("temperature", 0.7),
                do_sample=True
            )
            return response
        
        # Run the synchronous function in a thread pool
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, _generate)
        
        # Approximate token count
        tokens_used = len(prompt.split()) + len(response_text.split())
        
        return LLMResponse(
            text=response_text,
            model_name=self.name,
            tokens_used=tokens_used,
            metadata={"model": model_name}
        )
    
    async def count_tokens(self, text: str) -> int:
        """Approximate token count for Hugging Face models."""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4


class CohereInterface(ModelInterface):
    """Interface for Cohere models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import cohere
            self.client = cohere.AsyncClient(api_key=config.api_key)
        except ImportError:
            raise ImportError("Cohere package is required for Cohere models. Install with 'pip install cohere'")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using a Cohere model."""
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        response = await self.client.generate(
            prompt=prompt,
            model=params.get("model", self.config.name),
            max_tokens=params.get("max_tokens", 100),
            temperature=params.get("temperature", 0.7),
        )
        
        return LLMResponse(
            text=response.generations[0].text,
            model_name=self.name,
            tokens_used=response.meta.billed_units.input_tokens + response.meta.billed_units.output_tokens,
            metadata={"response": response}
        )
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using Cohere's tokenizer."""
        # Simple approximation: 1 token ≈ 4 characters
        # For more accurate counting, you would need to use Cohere's tokenizer
        return len(text) // 4


class GrokInterface(ModelInterface):
    """Interface for Grok models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import requests
            import aiohttp
            self.api_key = config.api_key
            self.base_url = config.base_url or "https://api.grok.ai/v1"
        except ImportError:
            raise ImportError("Aiohttp package is required for Grok models. Install with 'pip install aiohttp'")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using a Grok model."""
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": params.get("model", self.config.name),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": params.get("temperature", 0.7),
                "max_tokens": params.get("max_tokens", 1000),
            }
            
            async with session.post(f"{self.base_url}/chat/completions", 
                                  headers=headers, 
                                  json=payload) as response:
                result = await response.json()
                
                if "error" in result:
                    raise Exception(f"Grok API error: {result['error']}")
                
                return LLMResponse(
                    text=result["choices"][0]["message"]["content"],
                    model_name=self.name,
                    tokens_used=result.get("usage", {}).get("total_tokens", 0),
                    metadata={"response": result}
                )
    
    async def count_tokens(self, text: str) -> int:
        """Approximate token count for Grok models."""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4


class ReplicateInterface(ModelInterface):
    """Interface for Replicate models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import replicate
            self.client = replicate.Client(api_token=config.api_key)
        except ImportError:
            raise ImportError("Replicate package is required for Replicate models. Install with 'pip install replicate'")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using a Replicate model."""
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        # Use asyncio to run the synchronous API call in a thread pool
        import asyncio
        import functools
        
        # Define a synchronous function to call the Replicate API
        def _generate():
            model_path = params.get("model_path", "meta/llama-2-70b-chat")
            inputs = {
                "prompt": prompt,
                "temperature": params.get("temperature", 0.7),
                "max_new_tokens": params.get("max_tokens", 500),
                "system_prompt": params.get("system_prompt", "You are a helpful assistant.")
            }
            
            # Run the model
            output = self.client.run(model_path, input=inputs)
            # Replicate returns a generator, we need to join the outputs
            return "".join(output)
        
        # Run the synchronous function in a thread pool
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, _generate)
        
        # Approximate token count
        tokens_used = len(prompt.split()) + len(response_text.split())
        
        return LLMResponse(
            text=response_text,
            model_name=self.name,
            tokens_used=tokens_used,
            metadata={"model": params.get("model_path")}
        )
    
    async def count_tokens(self, text: str) -> int:
        """Approximate token count for Replicate models."""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4


def create_model_interface(config: ModelConfig) -> ModelInterface:
    """Factory function to create the appropriate model interface based on the provider."""
    if config.provider.lower() == "openai":
        return OpenAIInterface(config)
    elif config.provider.lower() == "anthropic":
        return AnthropicInterface(config)
    elif config.provider.lower() == "local":
        return LocalModelInterface(config)
    elif config.provider.lower() == "huggingface":
        return HuggingFaceInterface(config)
    elif config.provider.lower() == "cohere":
        return CohereInterface(config)
    elif config.provider.lower() == "grok":
        return GrokInterface(config)
    elif config.provider.lower() == "replicate":
        return ReplicateInterface(config)
    else:
        raise ValueError(f"Unsupported model provider: {config.provider}")