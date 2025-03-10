"""ReAct integration module for Agnassan tools.

This module provides integration between the tools registry and the ReAct reasoning system,
allowing language models to use registered tools as actions during reasoning.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable

from ..react import Action, ReActReasoning
from .index import ToolRegistry, list_tools, execute_tool

# Set up logging
logger = logging.getLogger("agnassan.tools.react_integration")


async def tool_executor(name: str, *args, **kwargs) -> Any:
    """Async wrapper for executing tools from the registry.
    
    Args:
        name: The name of the tool to execute
        *args: Positional arguments to pass to the tool
        **kwargs: Keyword arguments to pass to the tool
        
    Returns:
        The result of the tool execution
    """
    try:
        # Convert positional args to kwargs based on the tool's required args
        tool_info = ToolRegistry.get(name)
        if not tool_info:
            return f"Error: Tool '{name}' not found"
            
        # Map positional args to required args
        if args and tool_info["required_args"]:
            for i, arg_name in enumerate(tool_info["required_args"]):
                if i < len(args):
                    kwargs[arg_name] = args[i]
        
        # Validate required arguments
        for arg_name in tool_info.get("required_args", []):
            if arg_name not in kwargs:
                return f"Error: Required argument '{arg_name}' missing for tool '{name}'"
        
        # Execute the tool asynchronously
        if asyncio.iscoroutinefunction(execute_tool):
            result = await execute_tool(name, **kwargs)
        else:
            # Run synchronous tool in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: execute_tool(name, **kwargs)
            )
        
        return result
    except Exception as e:
        logger.error(f"Error executing tool '{name}': {str(e)}")
        return f"Error: {str(e)}"


def create_tool_actions() -> List[Action]:
    """Create Action objects for all registered tools.
    
    Returns:
        A list of Action objects that can be used with ReAct reasoning
    """
    actions = []
    
    # Get all registered tools
    tools = list_tools()
    
    for tool in tools:
        # Fix closure issue by creating a factory function
        def create_wrapper(tool_name: str) -> Callable:
            async def tool_action_wrapper(*args, **kwargs) -> Any:
                return await tool_executor(tool_name, *args, **kwargs)
            return tool_action_wrapper
            
        # Create an Action object for the tool
        action = Action(
            name=tool["name"],
            function=create_wrapper(tool["name"]),  # Use the factory function
            description=tool["description"]
        )
        
        actions.append(action)
    
    return actions


def integrate_tools_with_react(react_reasoning: ReActReasoning) -> None:
    """Integrate all registered tools with a ReAct reasoning instance.
    
    Args:
        react_reasoning: The ReAct reasoning instance to integrate tools with
    """
    # Create actions for all registered tools
    tool_actions = create_tool_actions()
    
    # Register the actions with the ReAct reasoning instance
    for action in tool_actions:
        react_reasoning.register_action(action)
    
    logger.info(f"Integrated {len(tool_actions)} tools with ReAct reasoning")


def create_react_with_tools() -> ReActReasoning:
    """Create a ReAct reasoning instance with all registered tools.
    
    Returns:
        A ReAct reasoning instance with all registered tools as actions
    """
    # Create a new ReAct reasoning instance
    react = ReActReasoning()
    
    # Integrate tools with the ReAct instance
    integrate_tools_with_react(react)
    
    return react