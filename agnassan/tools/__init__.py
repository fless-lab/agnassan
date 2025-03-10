"""Tools package for Agnassan.

This package provides various tools that enhance the capabilities of language models
by providing additional functionalities like web scraping, data processing, code generation,
image analysis, and more.
"""

from .index import ToolRegistry, register_tool, get_tool, list_tools, execute_tool

# Import all tool modules to register their tools
from . import web_tools
from . import data_tools
from . import image_tools
from . import code_tools
from . import nlp_tools

# Import the ReAct integration module
try:
    from . import react_integration
except ImportError:
    # This might happen if the react module is not available
    pass

__all__ = ['ToolRegistry', 'register_tool', 'get_tool', 'list_tools', 'execute_tool', 'react_integration']