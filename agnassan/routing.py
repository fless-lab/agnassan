"""Routing module for Agnassan.

This module handles the dynamic routing of queries to appropriate models.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import re
import asyncio
import logging

from .config import AgnassanConfig, ModelConfig
from .models import ModelInterface, LLMResponse, create_model_interface
from .reasoning import ReasoningEngine, ReasoningTechnique


class QueryClassifier:
    """Classifies queries to determine the most appropriate model."""
    
    def __init__(self):
        # Define patterns for different query types
        self.patterns = {
            "coding": r"(code|program|function|algorithm|script|debug|error|exception|syntax|compile)",
            "math": r"(math|equation|calculate|solve|formula|computation|arithmetic|algebra|calculus)",
            "creative": r"(creative|story|poem|imagine|fiction|design|art|music|novel|write)",
            "general_knowledge": r"(what is|who is|when did|where is|how does|explain|describe|define)",
            "complex_reasoning": r"(why|analyze|evaluate|compare|contrast|critique|implications|consequences)",
            "long_context": r"(document|summarize|article|book|chapter|text|transcript|conversation)",
            "vision": r"(image|picture|photo|screenshot|diagram|chart|graph|visual|see|look|view)",
            "multimodal": r"(website|mockup|design|ui|ux|interface|layout|document analysis|extract from image)"
        }
    
    def classify(self, query: str) -> Dict[str, float]:
        """Classify a query into different task types with confidence scores."""
        query = query.lower()
        scores = {}
        
        for task_type, pattern in self.patterns.items():
            matches = re.findall(pattern, query)
            score = len(matches) / max(len(query.split()), 1)  # Normalize by query length
            scores[task_type] = min(score * 2, 1.0)  # Cap at 1.0
        
        # Ensure all task types have a minimum score
        for task_type in self.patterns.keys():
            if task_type not in scores or scores[task_type] < 0.1:
                scores[task_type] = 0.1
        
        return scores


class ModelSelector:
    """Selects the most appropriate model(s) for a given query."""
    
    def __init__(self, config: AgnassanConfig):
        self.config = config
        self.classifier = QueryClassifier()
        self.logger = logging.getLogger("agnassan.routing")
    
    def select_models(self, query: str, max_models: int = 2) -> List[str]:
        """Select the most appropriate models for the query."""
        # Classify the query
        task_scores = self.classifier.classify(query)
        self.logger.debug(f"Query classification: {task_scores}")
        
        # Calculate model scores based on their strengths and the query classification
        model_scores = {}
        for model_config in self.config.models:
            score = 0.0
            for strength in model_config.strengths:
                if strength in task_scores:
                    score += task_scores[strength]
            
            # Adjust score based on cost (prefer free models unless paid ones are significantly better)
            if model_config.cost_per_token > 0:
                score *= 0.8  # Slight penalty for paid models
            
            model_scores[model_config.name] = score
        
        self.logger.debug(f"Model scores: {model_scores}")
        
        # Select the top N models
        selected_models = sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)[:max_models]
        
        # Always include the default model if it's not already selected
        if self.config.default_model not in selected_models:
            selected_models.append(self.config.default_model)
            if len(selected_models) > max_models:
                # Remove the lowest-scoring model that isn't the default
                non_default_models = [m for m in selected_models if m != self.config.default_model]
                lowest_model = min(non_default_models, key=lambda x: model_scores.get(x, 0))
                selected_models.remove(lowest_model)
        
        return selected_models


class ReasoningSelector:
    """Selects the most appropriate reasoning technique for a query."""
    
    def __init__(self, reasoning_engine: ReasoningEngine):
        self.reasoning_engine = reasoning_engine
        self.patterns = {
            "chain_of_thought": r"(step by step|explain how|process|procedure|method|approach)",
            "tree_of_thought": r"(different perspectives|multiple approaches|alternatives|various ways|consider)",
            "iterative_refinement": r"(improve|refine|enhance|optimize|better|perfect|polish)",
            "meta_critique": r"(critique|evaluate|assess|review|analyze|examine|judge)",
            "parallel_thought_chains": r"(combine approaches|multiple techniques|parallel thinking|diverse reasoning|synthesis|comprehensive analysis)",
            "iterative_loops": r"(progressive refinement|sequential improvement|iterative process|build upon|loop|cycle|repeated enhancement)"
        }
    
    def select_technique(self, query: str) -> str:
        """Select the most appropriate reasoning technique for the query."""
        query = query.lower()
        scores = {}
        
        for technique, pattern in self.patterns.items():
            matches = re.findall(pattern, query)
            score = len(matches) / max(len(query.split()), 1)  # Normalize by query length
            scores[technique] = score
        
        # Select the technique with the highest score, default to chain_of_thought
        if not scores or max(scores.values()) < 0.1:
            return "chain_of_thought"  # Default technique
        
        return max(scores.items(), key=lambda x: x[1])[0]


class Router:
    """Main router for Agnassan that orchestrates models and reasoning techniques."""
    
    def __init__(self, config: AgnassanConfig):
        self.config = config
        self.model_selector = ModelSelector(config)
        self.reasoning_engine = ReasoningEngine()
        self.reasoning_selector = ReasoningSelector(self.reasoning_engine)
        self.model_interfaces = {}
        self.logger = logging.getLogger("agnassan.router")
        
        # Import vision and multimodal modules only when needed
        try:
            from .vision import create_vision_model_interface
            from .multimodal import create_multimodal_processor, create_website_generator, create_document_analyzer, create_chart_analyzer
            self.has_vision_support = True
        except ImportError:
            self.logger.warning("Vision and multimodal support not available. Install required dependencies.")
            self.has_vision_support = False
    
    def _get_model_interface(self, model_name: str) -> ModelInterface:
        """Get or create a model interface for the specified model."""
        if model_name not in self.model_interfaces:
            model_config = self.config.get_model_config(model_name)
            if not model_config:
                raise ValueError(f"Model {model_name} not found in configuration")
            
            self.model_interfaces[model_name] = create_model_interface(model_config)
        
        return self.model_interfaces[model_name]
    
    def _get_vision_model_interface(self, model_name: str):
        """Get or create a vision model interface for the specified model."""
        if not self.has_vision_support:
            raise ValueError("Vision support not available. Install required dependencies.")
        
        from .vision import create_vision_model_interface
        
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        return create_vision_model_interface(model_config)
    
    def _is_multimodal_query(self, query: str) -> bool:
        """Determine if a query requires multimodal capabilities."""
        classifier = self.model_selector.classifier
        scores = classifier.classify(query)
        return scores.get("vision", 0) > 0.3 or scores.get("multimodal", 0) > 0.3
    
    async def route_query(self, query: str, strategy: str = "auto", image_path: str = None) -> LLMResponse:
        """Route a query to the appropriate model(s) and apply reasoning techniques.
        
        Args:
            query: The text query to process
            strategy: The routing strategy to use (default, auto, parallel)
            image_path: Optional path to an image for multimodal queries
        """
        self.logger.info(f"Routing query with strategy: {strategy}")
        
        # Check if this is a multimodal query (either explicitly via image_path or detected from query text)
        is_multimodal = image_path is not None or self._is_multimodal_query(query)
        
        if is_multimodal and not self.has_vision_support:
            self.logger.warning("Multimodal query detected but vision support not available")
            is_multimodal = False
        
        if is_multimodal:
            self.logger.info("Routing multimodal query to vision-capable model")
            # For multimodal queries, we need to select models that support vision
            from .vision import create_vision_model_interface
            from .multimodal import create_multimodal_processor
            
            # Prioritize open-source vision-capable models
            # First, check for models configured with open_source or local provider and vision capabilities
            open_source_vision_models = []
            for model_config in self.config.models:
                if model_config.provider.lower() in ["open_source", "local"] and \
                   model_config.parameters.get("vision_model_type") is not None:
                    open_source_vision_models.append(model_config.name)
            
            # If no explicitly configured open-source vision models, use any local model
            # We'll configure it with CLIP or other vision capabilities
            if not open_source_vision_models:
                for model_config in self.config.models:
                    if model_config.provider.lower() in ["open_source", "local"]:
                        open_source_vision_models.append(model_config.name)
            
            # As a last resort, check for commercial vision models
            commercial_vision_models = ["gpt-4o", "claude-3-opus"]
            
            # Find the first available vision-capable model in our config
            vision_model_name = None
            
            # First try open-source models
            for model_name in open_source_vision_models:
                if self.config.get_model_config(model_name):
                    vision_model_name = model_name
                    self.logger.info(f"Using open-source vision model: {model_name}")
                    break
            
            # If no open-source models available, try commercial models
            if not vision_model_name and not self.config.parameters.get("open_source_only", False):
                for model_name in commercial_vision_models:
                    if self.config.get_model_config(model_name):
                        vision_model_name = model_name
                        self.logger.info(f"Using commercial vision model: {model_name}")
                        break
            
            if not vision_model_name:
                self.logger.warning("No vision-capable models found in configuration")
                # Fall back to regular text processing
                is_multimodal = False
            else:
                # Process the multimodal query
                vision_model = self._get_vision_model_interface(vision_model_name)
                multimodal_processor = create_multimodal_processor(self.config.get_model_config(vision_model_name))
                
                if image_path:
                    return await multimodal_processor.process_text_and_image(query, image_path, vision_model)
                else:
                    # If no image path was provided but the query seems multimodal,
                    # we'll just use the vision-capable model for text processing
                    return await vision_model.generate(query)
        
        # Regular text query processing
        if strategy == "default":
            # Use only the default model with chain of thought reasoning
            model_interface = self._get_model_interface(self.config.default_model)
            return await self.reasoning_engine.apply_technique("chain_of_thought", query, model_interface)
        
        elif strategy == "parallel":
            # Use all configured models in parallel
            model_names = [model.name for model in self.config.models]
            model_interfaces = [self._get_model_interface(name) for name in model_names]
            
            # Create a parallel reasoning technique with all models
            parallel_technique = self.reasoning_engine.techniques.get("parallel_reasoning")
            if not parallel_technique:
                from .reasoning import ParallelReasoning
                parallel_technique = ParallelReasoning(model_interfaces)
                self.reasoning_engine.register_technique(parallel_technique)
            
            # Use the default model to synthesize the results
            default_model = self._get_model_interface(self.config.default_model)
            return await parallel_technique.apply(query, default_model)
        
        else:  # "auto" or any other strategy
            # Select the most appropriate models and reasoning technique
            selected_model_names = self.model_selector.select_models(query)
            reasoning_technique = self.reasoning_selector.select_technique(query)
            
            self.logger.info(f"Selected models: {selected_model_names}")
            self.logger.info(f"Selected reasoning technique: {reasoning_technique}")
            
            if len(selected_model_names) == 1:
                # Use a single model with the selected reasoning technique
                model_interface = self._get_model_interface(selected_model_names[0])
                
                # Special handling for advanced reasoning techniques that need the reasoning engine
                if reasoning_technique in ["parallel_thought_chains", "iterative_loops"]:
                    technique = self.reasoning_engine.techniques[reasoning_technique]
                    return await technique.apply(query, model_interface, self.reasoning_engine, **kwargs)
                else:
                    return await self.reasoning_engine.apply_technique(reasoning_technique, query, model_interface)
            else:
                # Get model interfaces for all selected models
                model_interfaces = [self._get_model_interface(name) for name in selected_model_names]
                primary_model = self._get_model_interface(selected_model_names[0])
                
                # Check if we should use one of our advanced reasoning techniques
                if reasoning_technique in ["parallel_thought_chains", "iterative_loops"]:
                    # Use the advanced technique with the primary model, but pass the reasoning engine
                    technique = self.reasoning_engine.techniques[reasoning_technique]
                    return await technique.apply(query, primary_model, self.reasoning_engine, **kwargs)
                elif reasoning_technique == "react":
                    # For ReAct, just use the primary model
                    return await self.reasoning_engine.apply_technique(reasoning_technique, query, primary_model)
                else:
                    # For other techniques, use parallel reasoning across multiple models
                    from .reasoning import ParallelReasoning
                    parallel_technique = ParallelReasoning(model_interfaces)
                    return await parallel_technique.apply(query, primary_model)
    
    async def close(self) -> None:
        """Close all model interfaces and free resources."""
        # Implement any cleanup needed for model interfaces
        pass