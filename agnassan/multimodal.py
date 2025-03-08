"""Multimodal module for Agnassan.

This module provides capabilities for handling multiple modalities (text, images, etc.)
and integrating them for more advanced applications.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import os

from .config import ModelConfig
from .models import LLMResponse, ModelInterface
from .vision import VisionModelInterface, create_vision_model_interface, ImageProcessor


class MultimodalError(Exception):
    """Exception raised for multimodal-related errors."""
    pass


class MultimodalProcessor:
    """Processes multiple modalities for integrated analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger("agnassan.multimodal")
        self.image_processor = ImageProcessor()
    
    async def process_text_and_image(self, text: str, image_path: str, model: VisionModelInterface) -> LLMResponse:
        """Process text and image together using a vision-enabled model."""
        if not os.path.exists(image_path):
            raise MultimodalError(f"Image file not found: {image_path}")
        
        try:
            # Check if we're using an open-source vision model
            if model.config.provider.lower() in ["open_source", "local"]:
                self.logger.info(f"Using open-source vision model: {model.name}")
                # If the model doesn't have vision capabilities configured, try to add them
                if not hasattr(model, "generate_with_image"):
                    try:
                        from .open_vision import create_open_vision_model_interface
                        # Create a new model with vision capabilities
                        model = create_open_vision_model_interface(model.config)
                    except ImportError:
                        self.logger.warning("Failed to load open-source vision capabilities")
            
            return await model.generate_with_image(text, image_path)
        except Exception as e:
            self.logger.error(f"Failed to process text and image: {str(e)}")
            raise MultimodalError(f"Failed to process text and image: {str(e)}")


class WebsiteGenerator:
    """Generates website code from images and descriptions."""
    
    def __init__(self, model: VisionModelInterface):
        self.model = model
        self.multimodal_processor = MultimodalProcessor()
        self.logger = logging.getLogger("agnassan.multimodal.website")
    
    async def generate_from_mockup(self, mockup_path: str, description: str = "") -> LLMResponse:
        """Generate website code from a mockup image."""
        prompt = f"""Generate HTML, CSS, and JavaScript code for a website based on this mockup image.
        
        {description if description else 'Create a responsive website that matches this design as closely as possible.'}
        
        Please provide the complete code with proper structure, including:
        1. HTML structure with semantic tags
        2. CSS styling (preferably using modern CSS features)
        3. Basic JavaScript functionality if needed
        
        The code should be well-organized, accessible, and follow best practices."""
        
        return await self.multimodal_processor.process_text_and_image(prompt, mockup_path, self.model)
    
    async def analyze_ui_design(self, design_path: str) -> LLMResponse:
        """Analyze a UI design and provide insights."""
        prompt = """Analyze this UI design and provide detailed feedback on:
        
        1. Overall layout and composition
        2. Color scheme and visual hierarchy
        3. Typography and readability
        4. User experience considerations
        5. Accessibility concerns
        6. Suggestions for improvement
        
        Be specific and provide actionable recommendations."""
        
        return await self.multimodal_processor.process_text_and_image(prompt, design_path, self.model)


class DocumentAnalyzer:
    """Analyzes documents containing text and images."""
    
    def __init__(self, model: VisionModelInterface):
        self.model = model
        self.multimodal_processor = MultimodalProcessor()
        self.logger = logging.getLogger("agnassan.multimodal.document")
    
    async def extract_information(self, document_path: str, query: str = "") -> LLMResponse:
        """Extract information from a document image."""
        prompt = f"""Extract and organize the key information from this document.
        
        {query if query else 'Please identify all important details, including any text, tables, figures, and their relationships.'}
        
        Format the extracted information in a structured way."""
        
        return await self.multimodal_processor.process_text_and_image(prompt, document_path, self.model)
    
    async def summarize_document(self, document_path: str) -> LLMResponse:
        """Summarize the content of a document image."""
        prompt = """Provide a comprehensive summary of this document.
        
        Include the main points, key findings, and any important details.
        The summary should be concise but thorough, capturing the essence of the document."""
        
        return await self.multimodal_processor.process_text_and_image(prompt, document_path, self.model)


class ChartAnalyzer:
    """Analyzes charts and graphs in images."""
    
    def __init__(self, model: VisionModelInterface):
        self.model = model
        self.multimodal_processor = MultimodalProcessor()
        self.logger = logging.getLogger("agnassan.multimodal.chart")
    
    async def interpret_chart(self, chart_path: str) -> LLMResponse:
        """Interpret and explain a chart or graph."""
        prompt = """Analyze this chart/graph and provide a detailed interpretation.
        
        Please include:
        1. The type of chart/graph
        2. What the axes or dimensions represent
        3. Key trends, patterns, or outliers
        4. The main insights or conclusions that can be drawn
        5. Any limitations or potential misinterpretations
        
        Be as specific and quantitative as possible in your analysis."""
        
        return await self.multimodal_processor.process_text_and_image(prompt, chart_path, self.model)
    
    async def extract_data(self, chart_path: str) -> LLMResponse:
        """Extract numerical data from a chart or graph."""
        prompt = """Extract the numerical data represented in this chart/graph.
        
        Please provide:
        1. The approximate values for each data point or series
        2. A tabular representation of the data if possible
        3. Any labels, legends, or annotations present in the image
        
        Format the data in a structured way that could be used for further analysis."""
        
        return await self.multimodal_processor.process_text_and_image(prompt, chart_path, self.model)


def create_multimodal_processor(config: ModelConfig) -> MultimodalProcessor:
    """Create a multimodal processor based on the configuration."""
    return MultimodalProcessor()


def create_website_generator(config: ModelConfig) -> WebsiteGenerator:
    """Create a website generator with the specified model."""
    vision_model = create_vision_model_interface(config)
    return WebsiteGenerator(vision_model)


def create_document_analyzer(config: ModelConfig) -> DocumentAnalyzer:
    """Create a document analyzer with the specified model."""
    vision_model = create_vision_model_interface(config)
    return DocumentAnalyzer(vision_model)


def create_chart_analyzer(config: ModelConfig) -> ChartAnalyzer:
    """Create a chart analyzer with the specified model."""
    vision_model = create_vision_model_interface(config)
    return ChartAnalyzer(vision_model)