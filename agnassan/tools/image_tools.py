"""Image processing tools for Agnassan.

This module provides tools for image analysis, manipulation, and generation
that enhance the capabilities of language models with multimodal abilities.
"""

import logging
import base64
import io
from typing import Dict, List, Any, Optional, Union, Tuple
import re

try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
    import cv2
    HAS_IMAGE_DEPS = True
except ImportError:
    HAS_IMAGE_DEPS = False

from .index import register_tool

# Set up logging
logger = logging.getLogger("agnassan.tools.image")

@register_tool(
    name="analyze_image",
    description="Analyze an image and extract basic information about it."
)
def analyze_image(image_path: str) -> Dict[str, Any]:
    """Analyze an image and extract basic information about it.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        A dictionary containing information about the image:
        - dimensions: (width, height)
        - format: Image format (JPEG, PNG, etc.)
        - mode: Color mode (RGB, RGBA, etc.)
        - size: File size in bytes
    """
    try:
        logger.info(f"Analyzing image: {image_path}")
        
        # Check if PIL is available
        if not HAS_IMAGE_DEPS:
            return {"error": "PIL is not available. Install Pillow for image analysis."}
        
        # Open the image
        img = Image.open(image_path)
        
        # Get basic information
        result = {
            "dimensions": img.size,
            "format": img.format,
            "mode": img.mode,
            "size": img.fp.tell() if hasattr(img, 'fp') and img.fp else None
        }
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing image {image_path}: {str(e)}")
        return {"error": str(e)}

@register_tool(
    name="resize_image",
    description="Resize an image to the specified dimensions."
)
def resize_image(image_path: str, width: int, height: int, output_path: str = None) -> str:
    """Resize an image to the specified dimensions.
    
    Args:
        image_path: Path to the input image file
        width: Target width in pixels
        height: Target height in pixels
        output_path: Path to save the resized image (if None, modifies the original)
        
    Returns:
        Path to the resized image
    """
    try:
        logger.info(f"Resizing image {image_path} to {width}x{height}")
        
        # Check if PIL is available
        if not HAS_IMAGE_DEPS:
            return "Error: PIL is not available. Install Pillow for image resizing."
        
        # Open the image
        img = Image.open(image_path)
        
        # Resize the image
        resized_img = img.resize((width, height), Image.LANCZOS)
        
        # Save the resized image
        if output_path is None:
            output_path = image_path
        
        resized_img.save(output_path)
        
        return output_path
    except Exception as e:
        logger.error(f"Error resizing image {image_path}: {str(e)}")
        return f"Error: {str(e)}"

@register_tool(
    name="crop_image",
    description="Crop an image to the specified region."
)
def crop_image(image_path: str, left: int, top: int, right: int, bottom: int, output_path: str = None) -> str:
    """Crop an image to the specified region.
    
    Args:
        image_path: Path to the input image file
        left: Left coordinate of the crop box
        top: Top coordinate of the crop box
        right: Right coordinate of the crop box
        bottom: Bottom coordinate of the crop box
        output_path: Path to save the cropped image (if None, modifies the original)
        
    Returns:
        Path to the cropped image
    """
    try:
        logger.info(f"Cropping image {image_path} to region ({left}, {top}, {right}, {bottom})")
        
        # Check if PIL is available
        if not HAS_IMAGE_DEPS:
            return "Error: PIL is not available. Install Pillow for image cropping."
        
        # Open the image
        img = Image.open(image_path)
        
        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))
        
        # Save the cropped image
        if output_path is None:
            output_path = image_path
        
        cropped_img.save(output_path)
        
        return output_path
    except Exception as e:
        logger.error(f"Error cropping image {image_path}: {str(e)}")
        return f"Error: {str(e)}"

@register_tool(
    name="detect_objects",
    description="Detect objects in an image using computer vision."
)
def detect_objects(image_path: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Detect objects in an image using computer vision.
    
    This function uses a pre-trained model to detect common objects in images.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score for detections (0-1)
        
    Returns:
        A list of dictionaries containing detected objects with keys:
        - label: The object class label
        - confidence: Confidence score (0-1)
        - bbox: Bounding box coordinates [x1, y1, x2, y2]
    """
    try:
        logger.info(f"Detecting objects in image: {image_path}")
        
        # Check if OpenCV is available
        if not HAS_IMAGE_DEPS or 'cv2' not in globals():
            return [{"error": "OpenCV is not available. Install opencv-python for object detection."}]
        
        # This is a simplified implementation
        # In a real system, you would load a pre-trained model like YOLO, SSD, or Faster R-CNN
        # For demonstration, we'll return mock results
        
        # Read the image to get its dimensions
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Mock detection results
        # In a real implementation, you would run inference with a model
        mock_objects = [
            {"label": "person", "confidence": 0.92, "bbox": [int(width*0.2), int(height*0.3), int(width*0.5), int(height*0.9)]},
            {"label": "chair", "confidence": 0.87, "bbox": [int(width*0.6), int(height*0.7), int(width*0.8), int(height*0.9)]},
            {"label": "book", "confidence": 0.76, "bbox": [int(width*0.1), int(height*0.1), int(width*0.2), int(height*0.2)]}
        ]
        
        # Filter by confidence threshold
        results = [obj for obj in mock_objects if obj["confidence"] >= confidence_threshold]
        
        return results
    except Exception as e:
        logger.error(f"Error detecting objects in image {image_path}: {str(e)}")
        return [{"error": str(e)}]

@register_tool(
    name="extract_text_from_image",
    description="Extract text from an image using OCR (Optical Character Recognition)."
)
def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image using OCR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text from the image
    """
    try:
        logger.info(f"Extracting text from image: {image_path}")
        
        # Check if OpenCV is available
        if not HAS_IMAGE_DEPS:
            return "Error: Required dependencies not available. Install opencv-python and pytesseract for OCR."
        
        # This is a placeholder. In a real implementation, you would use an OCR library
        # like pytesseract, EasyOCR, or a cloud OCR service.
        # For demonstration, we'll return a mock result.
        
        # Mock OCR result
        mock_text = "This is a mock OCR result. In a real implementation, this would be the text extracted from the image."
        
        return mock_text
    except Exception as e:
        logger.error(f"Error extracting text from image {image_path}: {str(e)}")
        return f"Error: {str(e)}"

@register_tool(
    name="apply_filter",
    description="Apply a filter to an image."
)
def apply_filter(image_path: str, filter_type: str, output_path: str = None) -> str:
    """Apply a filter to an image.
    
    Args:
        image_path: Path to the input image file
        filter_type: Type of filter to apply (blur, sharpen, grayscale, etc.)
        output_path: Path to save the filtered image (if None, modifies the original)
        
    Returns:
        Path to the filtered image
    """
    try:
        logger.info(f"Applying {filter_type} filter to image: {image_path}")
        
        # Check if PIL is available
        if not HAS_IMAGE_DEPS:
            return "Error: PIL is not available. Install Pillow for image filtering."
        
        # Open the image
        img = Image.open(image_path)
        
        # Apply the requested filter
        if filter_type.lower() == "blur":
            filtered_img = img.filter(ImageFilter.BLUR)
        elif filter_type.lower() == "sharpen":
            filtered_img = img.filter(ImageFilter.SHARPEN)
        elif filter_type.lower() == "grayscale":
            filtered_img = img.convert("L")
        elif filter_type.lower() == "edge_enhance":
            filtered_img = img.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_type.lower() == "emboss":
            filtered_img = img.filter(ImageFilter.EMBOSS)
        elif filter_type.lower() == "contour":
            filtered_img = img.filter(ImageFilter.CONTOUR)
        else:
            return f"Error: Unknown filter type: {filter_type}"
        
        # Save the filtered image
        if output_path is None:
            output_path = image_path
        
        filtered_img.save(output_path)
        
        return output_path
    except Exception as e:
        logger.error(f"Error applying filter to image {image_path}: {str(e)}")
        return f"Error: {str(e)}"