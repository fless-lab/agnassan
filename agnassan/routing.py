"""Routing module for Agnassan.

This module handles the dynamic routing of queries to appropriate models.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import re
import asyncio
import logging
import json
import hashlib
from functools import lru_cache

from .config import AgnassanConfig, ModelConfig
from .models import ModelInterface, LLMResponse, create_model_interface
from .reasoning import ReasoningEngine, ReasoningTechnique


class QueryClassifier:
    """Classifies queries to determine the most appropriate model."""
    
    def __init__(self, config=None):
        # Define patterns for different query types
        self.patterns = {
            "coding": r"(code|program|function|algorithm|script|debug|error|exception|syntax|compile)",
            "math": r"(math|equation|calculate|solve|formula|computation|arithmetic|algebra|calculus)",
            "creative": r"(creative|story|poem|imagine|fiction|design|art|music|novel|write)",
            "general_knowledge": r"(what is|who is|when did|where is|how does|explain|describe|define)",
            "complex_reasoning": r"(why|analyze|evaluate|compare|contrast|critique|implications|consequences)",
            "long_context": r"(document|summarize|article|book|chapter|text|transcript|conversation)",
            "vision": r"(image|picture|photo|screenshot|diagram|chart|graph|visual|see|look|view)",
            "multimodal": r"(website|mockup|design|ui|ux|interface|layout|document analysis|extract from image)",
            "tool_usage": r"(search|find|look up|fetch|analyze data|process|format|extract|transform|web search|tokenize|parse|summarize webpage|resize image|crop image)"
        }
        
        self.config = config
        self.logger = logging.getLogger("agnassan.query-classifier")
        self._classification_cache = {}
        self._max_cache_size = 100
        
    def _get_query_hash(self, query: str) -> str:
        """Generate a hash for a query to use as cache key."""
        return hashlib.md5(query.encode()).hexdigest()
        
    def _pattern_based_classify(self, query: str) -> Dict[str, float]:
        """Classify a query using regex pattern matching."""
        query = query.lower()
        scores = {}
        
        # Standard pattern matching
        for task_type, pattern in self.patterns.items():
            matches = re.findall(pattern, query)
            score = len(matches) / max(len(query.split()), 1)  # Normalize by query length
            scores[task_type] = min(score * 2, 1.0)  # Cap at 1.0
        
        # Ensure all task types have a minimum score
        for task_type in self.patterns.keys():
            if task_type not in scores or scores[task_type] < 0.1:
                scores[task_type] = 0.1
        
        return scores
    
    async def _model_based_classify(self, query: str, model_interface) -> Dict[str, float]:
        """Classify a query using a lightweight model."""
        try:
            # Create a prompt that asks the model to analyze the query
            classification_prompt = f"""Analyze this query and determine which task types are most relevant.
            Available task types: coding, math, creative, general_knowledge, complex_reasoning, long_context, vision, multimodal
            
            Query: "{query}"
            
            Return only a JSON object with task types as keys and confidence scores (0.0-1.0) as values.
            Example: {{"coding": 0.8, "math": 0.6, "general_knowledge": 0.3, "creative": 0.1, "complex_reasoning": 0.4, "long_context": 0.1, "vision": 0.1, "multimodal": 0.1}}"""
            
            # Generate response from the model
            print("Using model for classification:", model_interface)
            response = await model_interface.generate(classification_prompt, max_tokens=100, temperature=0.3)
            print("Model classification response:", response.text)
            
            # Parse the response to extract scores
            try:
                # Try to parse as JSON
                response_text = response.text.strip()
                # Find JSON object in the response if it's not a clean JSON
                if not response_text.startswith('{'):
                    import re
                    json_match = re.search(r'\{(.*?)\}', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0)
                
                scores = json.loads(response_text)
                if isinstance(scores, dict) and all(isinstance(scores[k], (int, float)) for k in scores):
                    # Ensure all task types have a score
                    for task_type in self.patterns.keys():
                        if task_type not in scores or not isinstance(scores[task_type], (int, float)):
                            scores[task_type] = 0.1
                    
                    self.logger.info(f"Model-based classification: {scores}")
                    return scores
                else:
                    self.logger.warning(f"Invalid model response format: {response.text}")
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse model response as JSON: {response.text}")
            
            # If we reach here, something went wrong with the model or parsing
            return self._pattern_based_classify(query)
            
        except Exception as e:
            self.logger.error(f"Error using model for classification: {str(e)}")
            return self._pattern_based_classify(query)
    
    async def classify(self, query: str, model_interface=None) -> Dict[str, float]:
        """Classify a query into different task types with confidence scores.
        
        Args:
            query: The user query
            model_interface: Optional model interface to use for classification
            
        Returns:
            Dictionary of task types and confidence scores
        """
        # Check cache first
        query_hash = self._get_query_hash(query)
        if query_hash in self._classification_cache:
            self.logger.debug(f"Using cached classification for query: {query[:50]}...")
            return self._classification_cache[query_hash]
        

        print("Model interface for classify : ",model_interface)

        # If no model provided, fall back to pattern matching
        if model_interface is None:
            scores = self._pattern_based_classify(query)
        else:
            scores = await self._model_based_classify(query, model_interface)
        
        # Cache the result before returning
        if len(self._classification_cache) >= self._max_cache_size:
            # Remove a random item if cache is full
            self._classification_cache.pop(next(iter(self._classification_cache)))
        
        self._classification_cache[query_hash] = scores
        return scores

class ModelSelector:
    """Selects the most appropriate model(s) for a given query."""
    
    def __init__(self, config: AgnassanConfig):
        self.config = config
        self.classifier = QueryClassifier(config)
        self.logger = logging.getLogger("agnassan.routing")
        self.lightweight_model = None
    
    async def _get_lightweight_model(self):
        """Get or create a lightweight model interface for classification."""
        print('self lightweight ?',self.lightweight_model)
        if self.lightweight_model is None:
            lightweight_model_name = self.config.parameters.get("classification_model")
            if lightweight_model_name:
                try:
                    from .models import create_model_interface
                    model_config = self.config.get_model_config(lightweight_model_name)
                    if model_config:
                        self.lightweight_model = create_model_interface(model_config)
                        self.logger.info(f"Using {lightweight_model_name} for query classification")
                except Exception as e:
                    self.logger.warning(f"Error initializing classification model: {str(e)}")
        
        return self.lightweight_model
    
    async def select_models(self, query: str, max_models: int = 2) -> List[str]:
        """Select the most appropriate models for the query."""
        # Try to use a lightweight model for classification if configured
        classification_model = await self._get_lightweight_model()
        print("Classification : ",classification_model)
        # Classify the query
        task_scores = await self.classifier.classify(query, classification_model)
        self.logger.info(f"Query classification: {task_scores}")
        
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
            self.logger.info(f"Model {model_config.name} score: {score}")
        
        self.logger.info(f"Model scores: {model_scores}")
        
        # Select the top N models
        selected_models = sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)[:max_models]
        self.logger.info(f"Selected models: {selected_models}")
        
        # Always include the default model if it's not already selected
        if self.config.default_model not in selected_models:
            selected_models.append(self.config.default_model)
            self.logger.info(f"Added default model: {self.config.default_model}")
            if len(selected_models) > max_models:
                # Remove the lowest-scoring model that isn't the default
                non_default_models = [m for m in selected_models if m != self.config.default_model]
                lowest_model = min(non_default_models, key=lambda x: model_scores.get(x, 0))
                selected_models.remove(lowest_model)
                self.logger.info(f"Removed lowest scoring model: {lowest_model}")
        
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
        self.logger = logging.getLogger("agnassan.reasoning_selector")
        # Initialize the technique detection cache
        self._technique_detection_cache = {}
        self._max_cache_size = 100
    
    def _get_query_hash(self, query: str) -> str:
        """Generate a hash for a query to use as cache key."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def select_technique(self, query: str) -> str:
        """Select the most appropriate reasoning technique for the query using pattern matching."""
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
    
    async def select_technique_with_model(self, query: str, model_interface: Optional[ModelInterface] = None) -> List[str]:
        """Select the most appropriate reasoning techniques using a lightweight model with caching.
        
        Args:
            query: The user query
            model_interface: Optional model interface to use for detection
            
        Returns:
            List of recommended reasoning technique names
        """
        # Check cache first
        query_hash = self._get_query_hash(query)
        if query_hash in self._technique_detection_cache:
            self.logger.debug(f"Using cached reasoning techniques for query: {query[:50]}...")
            return self._technique_detection_cache[query_hash]
        
        # If no model provided or model is too expensive/slow, fall back to pattern matching
        if model_interface is None:
            return [self.select_technique(query)]
        
        try:
            # Create a prompt that asks the model to analyze the query
            techniques_prompt = f"""Analyze this query and determine which reasoning techniques would be most appropriate.
            Available techniques: chain_of_thought, tree_of_thought, iterative_refinement, meta_critique, react, parallel_thought_chains, iterative_loops
            
            Query: "{query}"
            
            Return only the names of the recommended techniques as a JSON array, e.g., ["chain_of_thought", "meta_critique"]"""
            
            # Generate response from the model
            response = await model_interface.generate(techniques_prompt, max_tokens=50, temperature=0.3)
            
            # Parse the response to extract technique names
            # Process the model response
            response_text = response.text.strip()
            
            # Find JSON array in the response if it's not a clean JSON
            if not response_text.startswith('['):
                import re
                json_match = re.search(r'\[(.*?)\]', response_text)
                if json_match:
                    response_text = json_match.group(0)
                else:
                    # If no JSON array found, try to extract technique names directly
                    techniques_names = ["chain_of_thought", "tree_of_thought", "iterative_refinement", 
                                      "meta_critique", "react", "parallel_thought_chains", "iterative_loops"]
                    found_techniques = []
                    for technique in techniques_names:
                        if technique.lower() in response_text.lower():
                            found_techniques.append(technique)
                    if found_techniques:
                        self.logger.info(f"Extracted techniques from text: {found_techniques}")
                        return found_techniques
                    else:
                        self.logger.warning("No techniques found in text, using fallback")
                        return [self.select_technique(query)]
            
            # Try to parse the JSON response
            techniques = None
            try:
                techniques = json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse model response as JSON: {response.text}")
                fallback_technique = self.select_technique(query)
                return [fallback_technique]
            
            # Process the techniques if JSON parsing was successful
            if isinstance(techniques, list) and all(isinstance(t, str) for t in techniques):
                # Validate that all techniques exist in our reasoning engine
                valid_techniques = [t for t in techniques if t in self.reasoning_engine.techniques]
                if valid_techniques:
                    self.logger.info(f"Model detected reasoning techniques: {valid_techniques}")
                
                # Cache the result before returning
                if len(self._technique_detection_cache) >= self._max_cache_size:
                    # Remove a random item if cache is full
                    self._technique_detection_cache.pop(next(iter(self._technique_detection_cache)))
                
                self._technique_detection_cache[query_hash] = valid_techniques
                return valid_techniques
            else:
                self.logger.warning("Model returned no valid techniques, using fallback")
                self.logger.warning(f"Invalid model response format: {response.text}")
                
            # If we reach here, something went wrong with the model or parsing
            fallback_technique = self.select_technique(query)
            return [fallback_technique]
            
        except Exception as e:
            self.logger.error(f"Error using model for technique detection: {str(e)}")
            fallback_technique = self.select_technique(query)
            return [fallback_technique]


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
    
    async def _is_multimodal_query(self, query: str) -> bool:
        """Determine if a query requires multimodal capabilities."""
        classifier = self.model_selector.classifier
        scores = await classifier.classify(query)
        return scores.get("vision", 0) > 0.3 or scores.get("multimodal", 0) > 0.3
    
    async def _detect_reasoning_techniques(self, query: str, model_interface: Optional[ModelInterface] = None) -> List[str]:
        """Detect appropriate reasoning techniques for a given query using model-based approach.
        
        Args:
            query: The user query
            model_interface: Optional model interface to use for detection
            
        Returns:
            List of recommended reasoning technique names
        """
        return await self.reasoning_selector.select_technique_with_model(query, model_interface)
    
    async def route_query(self, query: str, strategy: str = "auto", image_path: str = None, **kwargs) -> LLMResponse:
        """Route a query to the appropriate model(s) and apply reasoning techniques.
        
        Args:
            query: The text query to process
            strategy: The routing strategy to use (default, auto, parallel)
            image_path: Optional path to an image for multimodal queries
            **kwargs: Additional parameters to pass to the model
        """
        self.logger.info(f"Routing query with strategy: {strategy}")
        
        # Check if this is a multimodal query (either explicitly via image_path or detected from query text)
        is_multimodal = image_path is not None or await self._is_multimodal_query(query)
        print("Is multimodal ?",is_multimodal)
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
            
            # If no suitable model found, use the default model (which may not have vision capabilities)
            if not vision_model_name:
                vision_model_name = self.config.default_model
                self.logger.warning(f"No vision-capable model found, using default model: {vision_model_name}")
            
            # Create the vision model interface
            vision_model = self._get_vision_model_interface(vision_model_name)
            
            # Create the multimodal processor
            multimodal_processor = create_multimodal_processor(vision_model)
            
            # Process the multimodal query
            self.logger.info(f"Processing multimodal query with model: {vision_model_name}")
            
            # Detect reasoning techniques using a lightweight model if available
            lightweight_model_name = self.config.parameters.get("lightweight_model")
            reasoning_techniques = []
            
            if lightweight_model_name:
                try:
                    lightweight_model = self._get_model_interface(lightweight_model_name)
                    reasoning_techniques = await self._detect_reasoning_techniques(query, lightweight_model)
                except Exception as e:
                    self.logger.warning(f"Error using lightweight model for technique detection: {str(e)}")
            
            if not reasoning_techniques:
                reasoning_techniques = await self._detect_reasoning_techniques(query)
            
            self.logger.info(f"Selected reasoning techniques: {reasoning_techniques}")
            
            # Process the multimodal query with the selected reasoning techniques
            return await multimodal_processor.process_text_and_image(
                query, image_path, reasoning_techniques=reasoning_techniques, **kwargs
            )
        
        # For text-only queries, use the selected routing strategy
        if strategy == "default":
            # Use the default model
            model_name = self.config.default_model
            model = self._get_model_interface(model_name)
            
            # Detect reasoning techniques
            reasoning_techniques = await self._detect_reasoning_techniques(query, model)
            
            # Apply the selected reasoning techniques
            return await self.reasoning_engine.apply_techniques(query, model, reasoning_techniques, **kwargs)
        
        elif strategy == "auto":
            print("We are in auto strategy")
            # Select the most appropriate model(s) for the query
            selected_models = await self.model_selector.select_models(query, max_models=1)
            print("Selected models",selected_models)
            model_name = selected_models[0]
            model = self._get_model_interface(model_name)
            print("rounting model is ",model,model_name)
            # Detect reasoning techniques using a lightweight model if available
            lightweight_model_name = self.config.parameters.get("lightweight_model")
            reasoning_techniques = []
            
            if lightweight_model_name:
                try:
                    lightweight_model = self._get_model_interface(lightweight_model_name)
                    reasoning_techniques = await self._detect_reasoning_techniques(query, lightweight_model)
                except Exception as e:
                    self.logger.warning(f"Error using lightweight model for technique detection: {str(e)}")
            
            if not reasoning_techniques:
                reasoning_techniques = await self._detect_reasoning_techniques(query, model)
            
            self.logger.info(f"Selected model: {model_name}, reasoning techniques: {reasoning_techniques}")
            
            # Apply the selected reasoning techniques
            return await self.reasoning_engine.apply_techniques(query, model, reasoning_techniques, **kwargs)
        
        elif strategy == "parallel":
            print("we are in parralel strategy")
            # Select multiple models to use in parallel
            selected_models = await self.model_selector.select_models(query, max_models=2)
            self.logger.info(f"Selected models for parallel execution: {selected_models}")
            
            # Create model interfaces
            models = [self._get_model_interface(name) for name in selected_models]
            
            # Detect reasoning techniques using a lightweight model if available
            lightweight_model_name = self.config.parameters.get("lightweight_model")
            reasoning_techniques = []
            
            if lightweight_model_name:
                try:
                    lightweight_model = self._get_model_interface(lightweight_model_name)
                    reasoning_techniques = await self._detect_reasoning_techniques(query, lightweight_model)
                except Exception as e:
                    self.logger.warning(f"Error using lightweight model for technique detection: {str(e)}")
            
            if not reasoning_techniques:
                # Use the first model to detect reasoning techniques
                reasoning_techniques = await self._detect_reasoning_techniques(query, models[0])
            
            self.logger.info(f"Selected reasoning techniques: {reasoning_techniques}")
            
            # Execute queries in parallel
            tasks = [
                self.reasoning_engine.apply_techniques(query, model, reasoning_techniques, **kwargs)
                for model in models
            ]
            
            responses = await asyncio.gather(*tasks)
            
            # Select the best response based on a quality heuristic
            # For now, we'll use the response with the most tokens as a simple heuristic
            best_response = max(responses, key=lambda r: r.tokens_used)
            
            # Add metadata about the parallel execution
            best_response.metadata["parallel_execution"] = {
                "models_used": selected_models,
                "selected_response": best_response.model_name
            }
            
            return best_response
        
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")
    
    async def close(self):
        """Close all model interfaces and free resources."""
        close_tasks = []
        for model_interface in self.model_interfaces.values():
            if hasattr(model_interface, "close") and callable(model_interface.close):
                close_tasks.append(model_interface.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks)
        
        self.model_interfaces.clear()
        self.logger.info("Router closed and resources freed")