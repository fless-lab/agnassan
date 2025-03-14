"""Configuration module for Agnassan.

This module handles the configuration of language models and system settings.
"""

import os
import yaml
from typing import Dict, List, Optional, Any, Union


class ModelConfig:
    """Configuration for a language model.
    
    Attributes:
        name: The name of the model.
        provider: The provider of the model (e.g., 'openai', 'anthropic', 'local').
        api_key_env: The environment variable name for the API key.
        base_url: Optional base URL for API calls.
        parameters: Additional parameters for the model.
        strengths: List of task types this model excels at.
        cost_per_token: Cost per token for paid models.
    """
    
    def __init__(
        self,
        name: str,
        provider: str,
        api_key_env: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        strengths: Optional[List[str]] = None,
        cost_per_token: float = 0.0
    ):
        self.name = name
        self.provider = provider
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.base_url = base_url
        self.parameters = parameters or {}
        self.strengths = strengths or []
        self.cost_per_token = cost_per_token
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create a ModelConfig from a dictionary."""
        return cls(
            name=config_dict['name'],
            provider=config_dict['provider'],
            api_key_env=config_dict.get('api_key_env'),
            base_url=config_dict.get('base_url'),
            parameters=config_dict.get('parameters', {}),
            strengths=config_dict.get('strengths', []),
            cost_per_token=config_dict.get('cost_per_token', 0.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ModelConfig to a dictionary."""
        return {
            'name': self.name,
            'provider': self.provider,
            'api_key_env': self.api_key_env if hasattr(self, 'api_key_env') else None,
            'base_url': self.base_url,
            'parameters': self.parameters,
            'strengths': self.strengths,
            'cost_per_token': self.cost_per_token
        }


class AgnassanConfig:
    """Main configuration for Agnassan.
    
    Attributes:
        models: List of configured language models.
        default_model: The default model to use when no specific routing is needed.
        log_dir: Directory for storing logs.
        routing_strategy: Strategy for routing queries to models.
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        default_model: str,
        log_dir: str = "./logs",
        routing_strategy: str = "rule_based",
        parameters: Dict[str, Any] = None
    ):
        self.models = models
        self.default_model = default_model
        self.log_dir = log_dir
        self.routing_strategy = routing_strategy
        self.parameters = parameters or {}
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AgnassanConfig':
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract model registry if present
        model_registry = config_dict.get('model_registry', {})
        
        # Process model configurations and inject model registry reference
        models = []
        for model_dict in config_dict['models']:
            # If model_registry is referenced in parameters, replace with actual registry
            if 'parameters' in model_dict and 'model_registry' in model_dict['parameters']:
                if model_dict['parameters']['model_registry'] == '${model_registry}':
                    model_dict['parameters']['model_registry'] = model_registry
            models.append(ModelConfig.from_dict(model_dict))
        
        return cls(
            models=models,
            default_model=config_dict['default_model'],
            log_dir=config_dict.get('log_dir', './logs'),
            routing_strategy=config_dict.get('routing_strategy', 'rule_based'),
            parameters=config_dict.get('parameters', {})
        )
    
    def save_to_yaml(self, config_path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {
            'models': [model.to_dict() for model in self.models],
            'default_model': self.default_model,
            'log_dir': self.log_dir,
            'routing_strategy': self.routing_strategy,
            'parameters': self.parameters
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get the configuration for a specific model."""
        for model in self.models:
            if model.name == model_name:
                return model
        return None


def create_default_config() -> AgnassanConfig:
    """Create a default configuration with open-source models only."""
    models = [
        # Open-source models
        ModelConfig(
            name="mistral-7b",
            provider="local",
            parameters={
                "model_path": "./models/mistral-7b",
                "context_length": 4096,
                "temperature": 0.7,
                "vision_model_type": "clip"  # Enable vision capabilities with CLIP
            },
            strengths=["general_knowledge", "coding", "reasoning"],
            cost_per_token=0.0
        ),
        ModelConfig(
            name="phi-2",
            provider="local",
            parameters={
                "model_path": "./models/phi-2",
                "context_length": 2048,
                "temperature": 0.7
            },
            strengths=["reasoning", "math", "coding"],
            cost_per_token=0.0
        ),
        ModelConfig(
            name="llama-3-8b",
            provider="local",
            parameters={
                "model_path": "./models/llama-3-8b",
                "context_length": 8192,
                "temperature": 0.7,
                "vision_model_type": "captioning"  # Enable vision capabilities with image captioning
            },
            strengths=["general_knowledge", "reasoning", "coding", "creative"],
            cost_per_token=0.0
        ),
        ModelConfig(
            name="gemma-7b",
            provider="local",
            parameters={
                "model_path": "./models/gemma-7b",
                "context_length": 8192,
                "temperature": 0.7,
                "vision_model_type": "classification"  # Enable vision capabilities with image classification
            },
            strengths=["general_knowledge", "math", "reasoning", "vision"],
            cost_per_token=0.0
        ),
        ModelConfig(
            name="mpt-7b",
            provider="local",
            parameters={
                "model_path": "./models/mpt-7b",
                "context_length": 2048,
                "temperature": 0.7
            },
            strengths=["general_knowledge", "creative"],
            cost_per_token=0.0
        ),
        ModelConfig(
            name="falcon-7b",
            provider="local",
            parameters={
                "model_path": "./models/falcon-7b",
                "context_length": 2048,
                "temperature": 0.7
            },
            strengths=["general_knowledge", "summarization"],
            cost_per_token=0.0
        )
    ]
    
    # Default parameters for the configuration
    default_parameters = {
        "lightweight_model": "phi-2",  # Use phi-2 as the lightweight model for reasoning technique detection
        "open_source_only": True      # Default to using only open-source models
    }
    
    return AgnassanConfig(
        models=models,
        default_model="llama-3-8b",
        routing_strategy="rule_based",
        parameters=default_parameters
    )