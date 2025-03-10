# """Code generation and analysis tools for Agnassan.

# This module provides tools for code generation, analysis, and transformation
# that enhance the capabilities of language models for programming tasks.
# """

# import logging
# import re
# import ast
# import json
# import os
# from typing import Dict, List, Any, Optional, Union

# try:
#     import black
#     import isort
#     HAS_CODE_DEPS = True
# except ImportError:
#     HAS_CODE_DEPS = False

# from .index import register_tool

# # Set up logging
# logger = logging.getLogger("agnassan.tools.code")

# @register_tool(
#     name="format_code",
#     description="Format code according to style guidelines."
# )
# def format_code(code: str, language: str = "python") -> str:
#     """Format code according to style guidelines.
    
#     Args:
#         code: The code to format
#         language: The programming language of the code (default: python)
        
#     Returns:
#         The formatted code
#     """
#     try:
#         logger.info(f"Formatting {language} code")
        
#         if language.lower() == "python":
#             # Check if black and isort are available
#             if not HAS_CODE_DEPS:
#                 return "# Note: Code formatting libraries not available. Install black and isort for better formatting.\n" + code
            
#             # Format with isort first (import sorting)
#             sorted_code = isort.code(code)
            
#             # Then format with black
#             formatted_code = black.format_str(sorted_code, mode=black.Mode())
            
#             return formatted_code
#         elif language.lower() in ["javascript", "typescript", "js", "ts"]:
#             # This is a simple indentation formatter for JS/TS
#             # In a real implementation, you would use a proper formatter like prettier
#             lines = code.split('\n')
#             indent_level = 0
#             formatted_lines = []
            
#             for line in lines:
#                 stripped = line.strip()
                
#                 # Adjust indent level based on braces
#                 if stripped.endswith('}') and not stripped.startswith('{'):
#                     indent_level = max(0, indent_level - 1)
                
#                 # Add the line with proper indentation
#                 if stripped:
#                     formatted_lines.append('  ' * indent_level + stripped)
#                 else:
#                     formatted_lines.append('')
                
#                 # Adjust indent level for the next line
#                 if stripped.endswith('{'):
#                     indent_level += 1
            
#             return '\n'.join(formatted_lines)
#         else:
#             # For other languages, return the original code
#             return f"# Note: Formatting not supported for {language}\n" + code
#     except Exception as e:
#         logger.error(f"Error formatting code: {str(e)}")
#         return f"# Error formatting code: {str(e)}\n" + code

# @register_tool(
#     name="analyze_code",
#     description="Analyze code and provide insights about its structure and quality."
# )
# def analyze_code(code: str, language: str = "python") -> Dict[str, Any]:
#     """Analyze code and provide insights about its structure and quality.
    
#     Args:
#         code: The code to analyze
#         language: The programming language of the code (default: python)
        
#     Returns:
#         A dictionary containing analysis results
#     """
#     try:
#         logger.info(f"Analyzing {language} code")
        
#         result = {
#             "language": language,
#             "line_count": len(code.split('\n')),
#             "char_count": len(code),
#             "structure": {},
#             "metrics": {},
#             "warnings": []
#         }
        
#         if language.lower() == "python":
#             try:
#                 # Parse the code into an AST
#                 tree = ast.parse(code)
                
#                 # Count different types of nodes
#                 functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
#                 classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
#                 imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                
#                 result["structure"] = {
#                     "function_count": len(functions),
#                     "class_count": len(classes),
#                     "import_count": len(imports),
#                     "functions": [func.name for func in functions],
#                     "classes": [cls.name for cls in classes]
#                 }
                
#                 # Simple metrics
#                 result["metrics"] = {
#                     "average_function_length": sum(len(ast.unparse(func).split('\n')) for func in functions) / len(functions) if functions else 0,
#                     "complexity": len(list(ast.walk(tree))) / result["line_count"] if result["line_count"] > 0 else 0
#                 }
                
#                 # Simple warnings
#                 for func in functions:
#                     if len(ast.unparse(func).split('\n')) > 50:
#                         result["warnings"].append(f"Function '{func.name}' is too long (> 50 lines)")
                
#                 # Check for unused imports (simplified)
#                 imported_names = set()
#                 for imp in imports:
#                     if isinstance(imp, ast.Import):
#                         for name in imp.names:
#                             imported_names.add(name.name)
#                     elif isinstance(imp, ast.ImportFrom):
#                         for name in imp.names:
#                             if name.asname:
#                                 imported_names.add(name.asname)
#                             else:
#                                 imported_names.add(name.name)
                
#                 # Get all names in the code
#                 all_names = set(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
                
#                 # Find unused imports
#                 for name in imported_names:
#                     if name not in all_names and name != '*':
#                         result["warnings"].append(f"Potentially unused import: '{name}'")
            
#             except SyntaxError as e:
#                 result["warnings"].append(f"Syntax error: {str(e)}")
        
#         elif language.lower() in ["javascript", "typescript", "js", "ts"]:
#             # Simple JS/TS analysis
#             # Count functions and classes using regex (not perfect but works for simple cases)
#             function_matches = re.findall(r'function\s+([\w$]+)\s*\(', code)
#             arrow_function_matches = re.findall(r'const\s+([\w$]+)\s*=\s*\([^)]*\)\s*=>', code)
#             class_matches = re.findall(r'class\s+([\w$]+)', code)
#             import_matches = re.findall(r'import\s+.+\s+from\s+[\'"](.*)[\'"]', code)
            
#             result["structure"] = {
#                 "function_count": len(function_matches) + len(arrow_function_matches),
#                 "class_count": len(class_matches),
#                 "import_count": len(import_matches),
#                 "functions": function_matches + arrow_function_matches,
#                 "classes": class_matches
#             }
            
#             # Simple warnings
#             if code.count('console.log') > 5:
#                 result["warnings"].append("Excessive use of console.log statements")
            
#             if code.count('var ') > 0:
#                 result["warnings"].append("Using 'var' instead of 'const' or 'let'")
        
#         return result
#     except Exception as e:
#         logger.error(f"Error analyzing code: {str(e)}")
#         return {"error": str(e)}

# @register_tool(
#     name="generate_docstring",
#     description="Generate a docstring for a function or class."
# )
# def generate_docstring(code: str, style: str = "google") -> str:
#     """Generate a docstring for a function or class.
    
#     Args:
#         code: The function or class code to document
#         style: The docstring style (google, numpy, or sphinx)
        
#     Returns:
#         The code with added docstring
#     """
#     try:
#         logger.info(f"Generating {style} style docstring")
        
#         # This is a simplified implementation
#         # In a real system, you would use a more sophisticated approach
#         # or a dedicated library like docstring-parser
        
#         # Try to parse the code
#         try:
#             tree = ast.parse(code)
#         except SyntaxError as e:
#             return f"# Error parsing code: {str(e)}\n" + code
        
#         # Find the first function or class definition
#         for node in ast.walk(tree):
#             if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
#                 # Extract function/class name and arguments
#                 name = node.name
                
#                 if isinstance(node, ast.FunctionDef):
#                     # Get function arguments
#                     args = [arg.arg for arg in node.args.args]
#                     returns = None
                    
#                     # Try to determine return type from return statements
#                     for child in ast.walk(node):
#                         if isinstance(child, ast.Return) and child.value:
#                             returns = "value"
#                             break
                    
#                     # Generate docstring based on style
#                     if style.lower() == "google":
#                         docstring = f'"""Function to {name.replace("_", " ")}\n\n'
#                         if args:
#                             docstring += "Args:\n"
#                             for arg in args:
#                                 if arg != "self":
#                                     docstring += f"    {arg}: Description of {arg}\n"
#                         if returns:
#                             docstring += "\nReturns:\n    Description of return value\n"
#                         docstring += '"""'
                    
#                     elif style.lower() == "numpy":
#                         docstring = f'"""Function to {name.replace("_", " ")}\n\n'
#                         if args:
#                             docstring += "Parameters\n----------\n"
#                             for arg in args:
#                                 if arg != "self":
#                                     docstring += f"{arg} : type\n    Description of {arg}\n"
#                         if returns:
#                             docstring += "\nReturns\n-------\n    Description of return value\n"
#                         docstring += '"""'
                    
#                     elif style.lower() == "sphinx":
#                         docstring = f'"""Function to {name.replace("_", " ")}\n\n'
#                         if args:
#                             for arg in args:
#                                 if arg != "self":
#                                     docstring += f":param {arg}: Description of {arg}\n"
#                         if returns:
#                             docstring += ":return: Description of return value\n"
#                         docstring += '"""'
                    
#                 else:  # ClassDef
#                     # Generate class docstring
#                     if style.lower() == "google":
#                         docstring = f'"""Class to {name.replace("_", " ")}\n\n'
#                         docstring += "Attributes:\n    Add class attributes here\n"
#                         docstring += '"""'
#                     elif style.lower() == "numpy":
#                         docstring = f'"""Class to {name.replace("_", " ")}\n\n'
#                         docstring += "Attributes\n----------\n    Add class attributes here\n"
#                         docstring += '"""'
#                     elif style.lower() == "sphinx":
#                         docstring = f'"""Class to {name.replace("_", " ")}\n\n'
#                         docstring += ":ivar: Add class attributes here\n"
#                         docstring += '"""'
                
#                 # Insert the docstring into the code
#                 lines = code.split('\n')
#                 # Find the line after the function/class definition
#                 for i, line in enumerate(lines):
#                     if f"def {name}" in line or f"class {name}" in line:
#                         # Insert docstring after the definition line
#                         indent = len(line) - len(line.lstrip())
#                         docstring_lines = docstring.split('\n')
#                         # Indent all lines of the docstring
#                         docstring_indented = '\n'.join([' ' * indent + line if i > 0 else line 
#                                                 for i, line in enumerate(docstring_lines)])
#                         lines.insert(i + 1, ' ' * indent + docstring_indented)
#                         break
                
#                 return '\n'.join(lines)
        
#         # If no function or class found
#         return code
    
#     except Exception as e:
#         logger.error(f"Error generating docstring: {str(e)}")
#         return f"# Error generating docstring: {str(e)}\n" + code

# @register_tool(
#     name="extract_code_snippets",
#     description="Extract code snippets from text or markdown."
# )
# def extract_code_snippets(text: str, language: str = None) -> List[str]:
#     """Extract code snippets from text or markdown.
    
#     Args:
#         text: The text or markdown content to extract code from
#         language: Optional language filter (e.g., 'python', 'javascript')
        
#     Returns:
#         A list of extracted code snippets
#     """
#     try:
#         logger.info(f"Extracting code snippets{' for ' + language if language else ''}")
        
#         # Match markdown code blocks with ```language ... ``` format
#         code_block_pattern = r'```(?P<lang>\w*)\n(?P<code>[\s\S]*?)\n```'
#         code_blocks = re.findall(code_block_pattern, text)
        
#         # Match inline code blocks with `code` format
#         inline_code_pattern = r'`([^`]+)`'
#         inline_blocks = re.findall(inline_code_pattern, text)
        
#         # Filter by language if specified
#         if language:
#             code_blocks = [(lang, code) for lang, code in code_blocks 
#                           if lang.lower() == language.lower() or not lang]
        
#         # Combine results, prioritizing code blocks
#         snippets = [code for _, code in code_blocks]
        
#         # Add inline blocks if they look like code (simplified heuristic)
#         for block in inline_blocks:
#             if len(block.split()) > 1 and ('(' in block or '=' in block or ';' in block):
#                 snippets.append(block)
        
#         return snippets
#     except Exception as e:
#         logger.error(f"Error extracting code snippets: {str(e)}")
#         return [f"# Error: {str(e)}"]

# @register_tool(
#     name="generate_unit_test",
#     description="Generate unit tests for a given function."
# )
# def generate_unit_test(code: str, test_framework: str = "pytest") -> str:
#     """Generate unit tests for a given function.
    
#     Args:
#         code: The function code to test
#         test_framework: The test framework to use (pytest, unittest)
        
#     Returns:
#         Generated unit test code
#     """
#     try:
#         logger.info(f"Generating {test_framework} unit tests")
        
#         # Parse the code to extract function information
#         try:
#             tree = ast.parse(code)
#         except SyntaxError as e:
#             return f"# Error parsing code: {str(e)}\n"
        
#         # Find the first function definition
#         for node in ast.walk(tree):
#             if isinstance(node, ast.FunctionDef):
#                 function_name = node.name
#                 args = [arg.arg for arg in node.args.args if arg.arg != 'self']
                
#                 # Generate test based on framework
#                 if test_framework.lower() == "pytest":
#                     test_code = f"""import pytest

# # Import the function to test
# # from your_module import {function_name}

# def test_{function_name}_basic():
#     """
#     Test basic functionality of {function_name}
#     """
#     # Setup
#     # TODO: Set up test inputs here
    
#     # Execute
#     result = {function_name}({', '.join(['None' for _ in args]) if args else ''})
    
#     # Assert
#     # TODO: Add assertions here
#     assert result is not None


# def test_{function_name}_edge_cases():
#     """Test edge cases for {function_name}."""
#     # TODO: Implement edge case tests
#     pass


# def test_{function_name}_error_handling():
#     """Test error handling in {function_name}."""
#     # TODO: Implement error handling tests
#     pass
# """
#                 elif test_framework.lower() == "unittest":
#                     test_code = f"""import unittest

# # Import the function to test
# # from your_module import {function_name}

# class Test{function_name.title()}(unittest.TestCase):
    
#     def setUp(self):
#         # Setup code that runs before each test
#         pass
    
#     def tearDown(self):
#         # Cleanup code that runs after each test
#         pass
    
#     def test_basic_functionality(self):
#         # Test basic functionality
#         result = {function_name}({', '.join(['None' for _ in args]) if args else ''})
#         self.assertIsNotNone(result)
    
#     def test_edge_cases(self):
#         # Test edge cases
#         pass
    
#     def test_error_handling(self):
#         # Test error handling
#         pass


# if __name__ == '__main__':
#     unittest.main()
# """
#                 else:
#                     test_code = f"# Unsupported test framework: {test_framework}\n# Supported frameworks: pytest, unittest\n"
                
#                 return test_code
        
#         return "# No function found to generate tests for\n"
#     except Exception as e:
#         logger.error(f"Error generating unit tests: {str(e)}")
#         return f"# Error generating unit tests: {str(e)}\n"