"""Open Source Vision module for Agnassan.

This module provides vision capabilities using open-source models for processing and analyzing images.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import base64
import os
import logging
from io import BytesIO

try:
    from PIL import Image
    import numpy as np
    import torch
    from transformers import CLIPProcessor, CLIPModel, AutoFeatureExtractor, AutoModelForImageClassification
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    HAS_VISION_DEPS = True
except ImportError:
    HAS_VISION_DEPS = False

from .config import ModelConfig
from .models import LLMResponse, ModelInterface
from .vision import ImageProcessor, VisionModelInterface, VisionError


class OpenSourceVisionInterface(VisionModelInterface):
    """Base interface for open-source vision-enabled language models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.logger = logging.getLogger("agnassan.open_vision")
        self.vision_model = None
        self.vision_processor = None
        self.text_model = None
        self.text_tokenizer = None
        self._load_models()
    
    def _load_models(self):
        """Load the vision and text models."""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """Generate a response using an open-source vision model with an image."""
        raise NotImplementedError("Subclasses must implement this method")


class CLIPVisionInterface(OpenSourceVisionInterface):
    """Interface for CLIP-based vision models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def _load_models(self):
        """Load the CLIP model and processor."""
        try:
            # Get model path from config, with fallback to default if not specified
            model_id = self.config.parameters.get("vision_model_id", "clip-vit-base-patch32")
            model_registry = self.config.parameters.get("model_registry", {})
            
            # Look up the model path in the registry or use the model_id as a path
            model_path = model_registry.get(model_id, f"openai/{model_id}")
            self.logger.info(f"Loading CLIP model from: {model_path}")
            
            self.vision_model = CLIPModel.from_pretrained(model_path)
            self.vision_processor = CLIPProcessor.from_pretrained(model_path)
            
            # Load a text model for generating responses
            text_model_id = self.config.parameters.get("text_model_id")
            if text_model_id:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                text_model_path = model_registry.get(text_model_id, text_model_id)
                self.logger.info(f"Loading text model from: {text_model_path}")
                self.text_model = AutoModelForCausalLM.from_pretrained(text_model_path)
                self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            else:
                # Use the model interface's model if available
                from .models import LocalModelInterface
                if isinstance(self, LocalModelInterface):
                    self.text_model = self.model
                    self.text_tokenizer = self.tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {str(e)}")
            raise VisionError(f"Failed to load CLIP model: {str(e)}")
    
    async def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """Generate a response using CLIP for image understanding and a text model for response generation."""
        if not os.path.exists(image_path):
            raise VisionError(f"Image file not found: {image_path}")
        
        try:
            # Load and process the image
            image = self.image_processor.load_image(image_path)
            
            # Process image and text with CLIP
            inputs = self.vision_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
            outputs = self.vision_model(**inputs)
            
            # Get image-text similarity score
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            confidence = probs[0][0].item()
            
            # Extract image features for text generation
            image_features = outputs.image_embeds
            
            # If we have a text model, generate a response based on the image features and prompt
            if self.text_model and self.text_tokenizer:
                # Create a prompt that includes information about the image
                enhanced_prompt = f"Based on the image, {prompt}"
                inputs = self.text_tokenizer(enhanced_prompt, return_tensors="pt")
                
                # Generate text
                with torch.no_grad():
                    outputs = self.text_model.generate(
                        inputs.input_ids,
                        max_new_tokens=kwargs.get("max_tokens", 100),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True
                    )
                
                response_text = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the output
                if response_text.startswith(enhanced_prompt):
                    response_text = response_text[len(enhanced_prompt):].strip()
            else:
                # If no text model is available, return a simple response with the confidence score
                response_text = f"The image is relevant to the prompt with {confidence:.2%} confidence."
            
            return LLMResponse(
                text=response_text,
                model_name=self.name,
                tokens_used=len(prompt.split()) + len(response_text.split()),  # Approximate
                metadata={"vision": True, "confidence": confidence}
            )
        except Exception as e:
            self.logger.error(f"Error processing image with CLIP: {str(e)}")
            raise VisionError(f"Error processing image with CLIP: {str(e)}")


class ImageCaptioningInterface(OpenSourceVisionInterface):
    """Interface for image captioning models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def _load_models(self):
        """Load the image captioning model."""
        try:
            # Get model path from config, with fallback to default if not specified
            model_id = self.config.parameters.get("vision_model_id", "vit-gpt2-image-captioning")
            model_registry = self.config.parameters.get("model_registry", {})
            
            # Look up the model path in the registry or use the model_id as a path
            model_path = model_registry.get(model_id, f"nlpconnect/{model_id}")
            self.logger.info(f"Loading image captioning model from: {model_path}")
            
            self.vision_model = VisionEncoderDecoderModel.from_pretrained(model_path)
            self.vision_processor = ViTImageProcessor.from_pretrained(model_path)
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.vision_model.to("cuda")
        except Exception as e:
            self.logger.error(f"Failed to load image captioning model: {str(e)}")
            raise VisionError(f"Failed to load image captioning model: {str(e)}")
    
    async def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """Generate a caption for an image and incorporate it into the response."""
        if not os.path.exists(image_path):
            raise VisionError(f"Image file not found: {image_path}")
        
        try:
            # Load and process the image
            image = self.image_processor.load_image(image_path)
            
            # Prepare image for the model
            pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to("cuda")
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.vision_model.generate(
                    pixel_values,
                    max_length=kwargs.get("max_tokens", 50),
                    num_beams=5,
                    temperature=kwargs.get("temperature", 0.7),
                )
            
            # Decode the caption
            caption = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Combine the caption with the prompt to create a response
            response_text = f"Image caption: {caption}\n\nBased on the image, which shows {caption}, here's my response to your question:\n{prompt}"
            
            # If we have a text model, use it to generate a more coherent response
            if hasattr(self, 'text_model') and self.text_model:
                enhanced_prompt = f"The image shows: {caption}. {prompt}"
                inputs = self.text_tokenizer(enhanced_prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.text_model.generate(
                        inputs.input_ids,
                        max_new_tokens=kwargs.get("max_tokens", 100),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True
                    )
                
                response_text = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the output
                if response_text.startswith(enhanced_prompt):
                    response_text = response_text[len(enhanced_prompt):].strip()
            
            return LLMResponse(
                text=response_text,
                model_name=self.name,
                tokens_used=len(prompt.split()) + len(response_text.split()),  # Approximate
                metadata={"vision": True, "caption": caption}
            )
        except Exception as e:
            self.logger.error(f"Error generating caption: {str(e)}")
            raise VisionError(f"Error generating caption: {str(e)}")


class ImageClassificationInterface(OpenSourceVisionInterface):
    """Interface for image classification models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def _load_models(self):
        """Load the image classification model."""
        try:
            # Get model path from config, with fallback to default if not specified
            model_id = self.config.parameters.get("vision_model_id", "vit-base-patch16-224")
            model_registry = self.config.parameters.get("model_registry", {})
            
            # Look up the model path in the registry or use the model_id as a path
            model_path = model_registry.get(model_id, f"google/{model_id}")
            self.logger.info(f"Loading image classification model from: {model_path}")
            
            self.vision_model = AutoModelForImageClassification.from_pretrained(model_path)
            self.vision_processor = AutoFeatureExtractor.from_pretrained(model_path)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.vision_model.to("cuda")
        except Exception as e:
            self.logger.error(f"Failed to load image classification model: {str(e)}")
            raise VisionError(f"Failed to load image classification model: {str(e)}")
    
    async def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> LLMResponse:
        """Classify an image and incorporate the classification into the response."""
        if not os.path.exists(image_path):
            raise VisionError(f"Image file not found: {image_path}")
        
        try:
            # Load and process the image
            image = self.image_processor.load_image(image_path)
            
            # Prepare image for the model
            inputs = self.vision_processor(image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Perform classification
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
            
            # Get the predicted class
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            
            # Get the class label if available
            if hasattr(self.vision_model.config, 'id2label'):
                predicted_class = self.vision_model.config.id2label[predicted_class_idx]
            else:
                predicted_class = f"Class {predicted_class_idx}"
            
            # Create a response that incorporates the classification
            response_text = f"The image appears to be: {predicted_class}\n\nBased on this classification, here's my response to your question:\n{prompt}"
            
            # If we have a text model, use it to generate a more coherent response
            if hasattr(self, 'text_model') and self.text_model:
                enhanced_prompt = f"The image shows: {predicted_class}. {prompt}"
                inputs = self.text_tokenizer(enhanced_prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.text_model.generate(
                        inputs.input_ids,
                        max_new_tokens=kwargs.get("max_tokens", 100),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True
                    )
                
                response_text = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the output
                if response_text.startswith(enhanced_prompt):
                    response_text = response_text[len(enhanced_prompt):].strip()
            
            return LLMResponse(
                text=response_text,
                model_name=self.name,
                tokens_used=len(prompt.split()) + len(response_text.split()),  # Approximate
                metadata={"vision": True, "classification": predicted_class}
            )
        except Exception as e:
            self.logger.error(f"Error classifying image: {str(e)}")
            raise VisionError(f"Error classifying image: {str(e)}")


def create_open_vision_model_interface(config: ModelConfig) -> VisionModelInterface:
    """Create an open-source vision model interface based on the model type."""
    model_type = config.parameters.get("vision_model_type", "clip")
    
    # Log the model type being used
    logging.getLogger("agnassan.open_vision").info(f"Creating open vision model interface of type: {model_type}")
    
    if model_type.lower() == "clip":
        return CLIPVisionInterface(config)
    elif model_type.lower() == "captioning":
        return ImageCaptioningInterface(config)
    elif model_type.lower() == "classification":
        return ImageClassificationInterface(config)
    else:
        logging.getLogger("agnassan.open_vision").warning(f"Unknown vision model type: {model_type}, defaulting to CLIP")
        return CLIPVisionInterface(config)
    else:
        raise ValueError(f"Unsupported open-source vision model type: {model_type}")