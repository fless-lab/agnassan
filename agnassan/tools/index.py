"""Tool registry and management for Agnassan.

This module provides a central registry for all tools available to language models
in the Agnassan system. It allows for dynamic registration, discovery, and execution
of tools that extend the capabilities of language models.
"""

from typing import Dict, List, Any, Callable, Optional, Union
import inspect
import logging
import functools

# Set up logging
logger = logging.getLogger("agnassan.tools")

class ToolRegistry:
    """A registry for tools that can be used by language models.
    
    This class manages the registration, retrieval, and execution of tools
    that extend the capabilities of language models in Agnassan.
    """
    
    _tools: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, description: str, function: Callable, 
                 required_args: List[str] = None, optional_args: Dict[str, Any] = None):
        """Register a tool in the registry.
        
        Args:
            name: Unique identifier for the tool
            description: Human-readable description of what the tool does
            function: The callable that implements the tool's functionality
            required_args: List of required argument names
            optional_args: Dictionary of optional arguments with their default values
        """
        if name in cls._tools:
            logger.warning(f"Tool '{name}' is already registered. Overwriting.")
        
        # Extract argument information if not provided
        if required_args is None or optional_args is None:
            sig = inspect.signature(function)
            if required_args is None:
                required_args = []
                for param_name, param in sig.parameters.items():
                    if param.default == inspect.Parameter.empty and param_name != 'self':
                        required_args.append(param_name)
            
            if optional_args is None:
                optional_args = {}
                for param_name, param in sig.parameters.items():
                    if param.default != inspect.Parameter.empty:
                        optional_args[param_name] = param.default
        
        cls._tools[name] = {
            "name": name,
            "description": description,
            "function": function,
            "required_args": required_args,
            "optional_args": optional_args
        }
        
        logger.info(f"Registered tool: {name}")
        return function  # Return the function to allow use as a decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name.
        
        Args:
            name: The name of the tool to retrieve
            
        Returns:
            The tool information dictionary or None if not found
        """
        return cls._tools.get(name)
    
    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        """List all registered tools.
        
        Returns:
            A list of tool information dictionaries
        """
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "required_args": info["required_args"],
                "optional_args": info["optional_args"]
            } for info in cls._tools.values()
        ]
    
    @classmethod
    def execute(cls, name: str, **kwargs) -> Any:
        """Execute a tool by name with the provided arguments.
        
        Args:
            name: The name of the tool to execute
            **kwargs: Arguments to pass to the tool function
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If the tool is not found or required arguments are missing
        """
        tool = cls.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        # Check for required arguments
        missing_args = [arg for arg in tool["required_args"] if arg not in kwargs]
        if missing_args:
            raise ValueError(f"Missing required arguments for tool '{name}': {missing_args}")
        
        # Add default values for optional arguments if not provided
        for arg, default in tool["optional_args"].items():
            if arg not in kwargs:
                kwargs[arg] = default
        
        # Execute the tool function
        try:
            logger.debug(f"Executing tool: {name}")
            return tool["function"](**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {str(e)}")
            raise

# Convenience functions that use the ToolRegistry
def register_tool(name: str, description: str, required_args: List[str] = None, 
                 optional_args: Dict[str, Any] = None):
    """Decorator to register a function as a tool.
    
    Args:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        required_args: List of required argument names
        optional_args: Dictionary of optional arguments with their default values
        
    Returns:
        A decorator function that registers the decorated function as a tool
    """
    def decorator(func):
        ToolRegistry.register(name, description, func, required_args, optional_args)
        return func
    return decorator

def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a tool by name.
    
    Args:
        name: The name of the tool to retrieve
        
    Returns:
        The tool information dictionary or None if not found
    """
    return ToolRegistry.get(name)

def list_tools() -> List[Dict[str, Any]]:
    """List all registered tools.
    
    Returns:
        A list of tool information dictionaries
    """
    return ToolRegistry.list()

def execute_tool(name: str, **kwargs) -> Any:
    """Execute a tool by name with the provided arguments.
    
    Args:
        name: The name of the tool to execute
        **kwargs: Arguments to pass to the tool function
        
    Returns:
        The result of the tool execution
    """
    return ToolRegistry.execute(name, **kwargs)