"""Vision module for Agnassan.

This module provides vision capabilities for processing and analyzing images.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import base64
import os
import logging
from io import BytesIO

try:
    from PIL import Image
    import numpy as np
    HAS_VISION_DEPS = True
except ImportError:
    HAS_VISION_DEPS = False

from .config import ModelConfig
from .models import LLMResponse, ModelInterface


class VisionError(Exception):
    """Exception raised for vision-related errors."""
    pass


class ImageProcessor:
    """Processes images for vision-enabled language models."""
    
    def __init__(self):
        self.logger = logging.getLogger("agnassan.vision")
        if not HAS_VISION_DEPS:
            self.logger.warning("Vision dependencies not installed. Install with 'pip install pillow numpy'")
    
    def validate_dependencies(self):
        """Check if required dependencies are installed."""
        if not HAS_VISION_DEPS:
            raise VisionError("Vision dependencies not installed. Install with 'pip install pillow numpy'")
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path."""
        self.validate_dependencies()
        
        if not os.path.exists(image_path):
            raise VisionError(f"Image file not found: {image_path}")
        
        try:
            return Image.open(image_path)
        except Exception as e:
            raise VisionError(f"Failed to load image: {str(e)}")
    
    def resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize an image while maintaining aspect ratio."""
        self.validate_dependencies()
        
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def encode_image_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """Encode an image as a base64 string."""
        self.validate_dependencies()
        
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def prepare_image_for_model(self, image_path: str, max_size: int = 1024) -> str:
        """Prepare an image for use with a vision-enabled model."""
        image = self.load_image(image_path)
        resized_image = self.resize_image(image, max_size)
        return self.encode_image_base64(resized_image)


class VisionModelInterface(ModelInterface):
    """Interface for vision-enabled language models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.image_processor = ImageProcessor()
    
    async def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """Generate a response from the language model with an image input."""
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIVisionInterface(VisionModelInterface):
    """Interface for OpenAI vision-enabled models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url
            )
        except ImportError:
            raise ImportError("OpenAI package is required for OpenAI models. Install with 'pip install openai'")
    
    async def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """Generate a response using an OpenAI vision model with an image."""
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        # Prepare the image
        base64_image = self.image_processor.prepare_image_for_model(image_path)
        
        # Create the message with text and image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]
        
        # Get model ID from config and look it up in the registry
        model_id = params.get("model_id", self.config.name)
        model_registry = params.get("model_registry", {})
        
        # Look up the model name in the registry or use the model_id as the name
        model_name = model_registry.get(model_id, model_id)
        
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 1000),
        )
        
        return LLMResponse(
            text=response.choices[0].message.content,
            model_name=self.name,
            tokens_used=response.usage.total_tokens,
            metadata={"response": response, "vision": True}
        )
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using an OpenAI model (without image)."""
        # Delegate to the parent class implementation
        return await super().generate(prompt, **kwargs)


class AnthropicVisionInterface(VisionModelInterface):
    """Interface for Anthropic vision-enabled models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=config.api_key
            )
        except ImportError:
            raise ImportError("Anthropic package is required for Anthropic models. Install with 'pip install anthropic'")
    
    async def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """Generate a response using an Anthropic vision model with an image."""
        params = self.config.parameters.copy()
        params.update(kwargs)
        
        # Prepare the image
        base64_image = self.image_processor.prepare_image_for_model(image_path)
        
        # Create the message with text and image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                ]
            }
        ]
        
        # Get model ID from config and look it up in the registry
        model_id = params.get("model_id", self.config.name)
        model_registry = params.get("model_registry", {})
        
        # Look up the model name in the registry or use the model_id as the name
        model_name = model_registry.get(model_id, model_id)
        
        response = await self.client.messages.create(
            model=model_name,
            messages=messages,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 1000),
        )
        
        return LLMResponse(
            text=response.content[0].text,
            model_name=self.name,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            metadata={"response": response, "vision": True}
        )
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using an Anthropic model (without image)."""
        # Delegate to the parent class implementation
        return await super().generate(prompt, **kwargs)


def create_vision_model_interface(config: ModelConfig) -> VisionModelInterface:
    """Create a vision model interface based on the provider."""
    provider = config.provider.lower()
    
    if provider == "openai":
        return OpenAIVisionInterface(config)
    elif provider == "anthropic":
        return AnthropicVisionInterface(config)
    elif provider == "open_source" or provider == "local":
        # Import the open-source vision interfaces
        try:
            from .open_vision import create_open_vision_model_interface
            return create_open_vision_model_interface(config)
        except ImportError:
            raise ImportError("Open-source vision models require additional dependencies. Install with 'pip install transformers torch'")
    else:
        raise ValueError(f"Unsupported vision model provider: {provider}")