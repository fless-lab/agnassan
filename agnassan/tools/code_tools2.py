# """Code generation and analysis tools for Agnassan.

# This module provides tools for code generation, analysis, and transformation
# that enhance the capabilities of language models for programming tasks.
# """

# import logging
# import re
# import ast
# import json
# import os
# import inspect
# from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# try:
#     import black
#     import isort
#     import pylint.lint
#     from radon.complexity import cc_visit
#     from radon.metrics import mi_visit
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
#         elif language.lower() in ["rust", "rs"]:
#             # Simple Rust formatter (in production, use rustfmt)
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
#                     formatted_lines.append('    ' * indent_level + stripped)
#                 else:
#                     formatted_lines.append('')
                
#                 # Adjust indent level for the next line
#                 if stripped.endswith('{'):
#                     indent_level += 1
            
#             return '\n'.join(formatted_lines)
#         elif language.lower() in ["go", "golang"]:
#             # Simple Go formatter (in production, use gofmt)
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
#                     formatted_lines.append('\t' * indent_level + stripped)
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
                
#                 # Advanced metrics if radon is available
#                 if HAS_CODE_DEPS:
#                     try:
#                         # Calculate cyclomatic complexity
#                         cc_results = cc_visit(code)
#                         avg_complexity = sum(cc.complexity for cc in cc_results) / len(cc_results) if cc_results else 0
                        
#                         # Calculate maintainability index
#                         mi_result = mi_visit(code, multi=True)
                        
#                         result["metrics"].update({
#                             "average_complexity": round(avg_complexity, 2),
#                             "maintainability_index": round(mi_result, 2) if isinstance(mi_result, (int, float)) else None,
#                             "complexity_details": [{"name": cc.name, "complexity": cc.complexity} for cc in cc_results]
#                         })
                        
#                         # Add warnings based on complexity
#                         for cc in cc_results:
#                             if cc.complexity > 10:
#                                 result["warnings"].append(f"High complexity ({cc.complexity}) in '{cc.name}'")
#                     except Exception as e:
#                         logger.warning(f"Error calculating advanced metrics: {str(e)}")
                
#                 # Basic metrics
#                 result["metrics"].update({
#                     "average_function_length": sum(len(ast.unparse(func).split('\n')) for func in functions) / len(functions) if functions else 0,
#                     "loc_per_function": result["line_count"] / len(functions) if functions else 0,
#                     "comment_ratio": len(re.findall(r'^\s*#.*$', code, re.MULTILINE)) / result["line_count"] if result["line_count"] > 0 else 0,
#                 })
                
#                 # Simple warnings
#                 for func in functions:
#                     if len(ast.unparse(func).split('\n')) > 50:
#                         result["warnings"].append(f"Function '{func.name}' is too long (> 50 lines)")
                    
#                     # Check for function arguments
#                     if len(func.args.args) > 5:
#                         result["warnings"].append(f"Function '{func.name}' has too many arguments ({len(func.args.args)})")
                
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
#             # Enhanced JS/TS analysis
#             function_matches = re.findall(r'function\s+([\w$]+)\s*\(', code)
#             arrow_function_matches = re.findall(r'const\s+([\w$]+)\s*=\s*\([^)]*\)\s*=>', code)
#             class_matches = re.findall(r'class\s+([\w$]+)', code)
#             import_matches = re.findall(r'import\s+.+\s+from\s+[\'"](.*)[\'"]', code)
            
#             # Find React components (simplified)
#             react_component_matches = re.findall(r'function\s+([\w$]+)\s*\([^)]*\)\s*{[^}]*return\s*\(\s*<', code) 
            
#             result["structure"] = {
#                 "function_count": len(function_matches) + len(arrow_function_matches),
#                 "class_count": len(class_matches),
#                 "import_count": len(import_matches),
#                 "functions": function_matches + arrow_function_matches,
#                 "classes": class_matches,
#                 "react_components": react_component_matches
#             }
            
#             # Calculate metrics
#             result["metrics"].update({
#                 "es6_ratio": (code.count('=>') + code.count('const ') + code.count('let ')) / result["line_count"] if result["line_count"] > 0 else 0,
#                 "comment_ratio": len(re.findall(r'^\s*\/\/.*$', code, re.MULTILINE)) / result["line_count"] if result["line_count"] > 0 else 0,
#             })
            
#             # Warnings
#             if code.count('console.log') > 5:
#                 result["warnings"].append("Excessive use of console.log statements")
            
#             if code.count('var ') > 0:
#                 result["warnings"].append("Using 'var' instead of 'const' or 'let'")
                
#             if code.count('== ') > 0:
#                 result["warnings"].append("Using loose equality (==) instead of strict equality (===)")
                
#             if code.count('try {') > 0 and code.count('catch (') == 0:
#                 result["warnings"].append("Try block without catch")
                
#             # Check for potentially complex functions (simple heuristic)
#             function_bodies = re.findall(r'function\s+[\w$]+\s*\([^)]*\)\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}', code)
#             for body in function_bodies:
#                 if body.count('if') + body.count('for') + body.count('while') + body.count('switch') > 5:
#                     result["warnings"].append("Complex function with many conditionals/loops")
        
#         elif language.lower() in ["rust", "rs"]:
#             # Simple Rust analysis
#             struct_matches = re.findall(r'struct\s+(\w+)', code)
#             enum_matches = re.findall(r'enum\s+(\w+)', code)
#             impl_matches = re.findall(r'impl(?:<[^>]*>)?\s+(\w+)', code)
#             fn_matches = re.findall(r'fn\s+(\w+)', code)
            
#             result["structure"] = {
#                 "struct_count": len(struct_matches),
#                 "enum_count": len(enum_matches),
#                 "impl_count": len(impl_matches),
#                 "function_count": len(fn_matches),
#                 "structs": struct_matches,
#                 "enums": enum_matches,
#                 "impls": impl_matches,
#                 "functions": fn_matches
#             }
            
#             # Rust-specific warnings
#             if code.count('unwrap()') > 3:
#                 result["warnings"].append("Excessive use of unwrap() which may cause panics")
                
#             if code.count('unsafe {') > 0:
#                 result["warnings"].append("Contains unsafe blocks")
                
#             if 'match ' in code and 'match ' in code and '_' not in code:
#                 result["warnings"].append("Match expression without default case (_)")
                
#             # Check for potential lifetime issues (simplified)
#             if "'" in code and "<'" in code and code.count("'static") == 0:
#                 result["warnings"].append("Uses lifetimes but no 'static - review for potential lifetime issues")
        
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
#                     # Get function arguments with type hints if available
#                     args = []
#                     for arg in node.args.args:
#                         arg_name = arg.arg
#                         # Try to get type annotation if available
#                         arg_type = ""
#                         if arg.annotation:
#                             try:
#                                 arg_type = ast.unparse(arg.annotation)
#                             except:
#                                 pass
#                         args.append((arg_name, arg_type))
                    
#                     # Try to determine return type from return statements and annotations
#                     returns = None
#                     return_type = ""
#                     if node.returns:
#                         try:
#                             return_type = ast.unparse(node.returns)
#                         except:
#                             pass
                    
#                     # Check return statements
#                     for child in ast.walk(node):
#                         if isinstance(child, ast.Return) and child.value:
#                             returns = "value"
#                             break
                    
#                     # Generate docstring based on style
#                     if style.lower() == "google":
#                         docstring = f'"""Function to {name.replace("_", " ")}.
                        
# This function {name.replace("_", " ")}.

# Args:
# '
#                         for arg_name, arg_type in args:
#                             if arg_name != "self" and arg_name != "cls":
#                                 docstring += f"    {arg_name}{f' ({arg_type})' if arg_type else ''}: Description of {arg_name}\n"
                        
#                         if returns:
#                             docstring += f"\nReturns:\n    {return_type if return_type else 'Any'}: Description of return value\n"
                        
#                         if any('raise' in ast.unparse(expr) for expr in ast.walk(node) if isinstance(expr, ast.Expr)):
#                             docstring += "\nRaises:\n    Exception: Description of when this error is raised\n"
                            
#                         docstring += '"""'
                    
#                     elif style.lower() == "numpy":
#                         docstring = f'"""Function to {name.replace("_", " ")}.

# This function {name.replace("_", " ")}.

# Parameters
# ----------
# '
#                         for arg_name, arg_type in args:
#                             if arg_name != "self" and arg_name != "cls":
#                                 docstring += f"{arg_name} : {arg_type if arg_type else 'type'}\n    Description of {arg_name}\n"
                        
#                         if returns:
#                             docstring += f"\nReturns\n-------\n{return_type if return_type else 'Any'}\n    Description of return value\n"
                            
#                         if any('raise' in ast.unparse(expr) for expr in ast.walk(node) if isinstance(expr, ast.Expr)):
#                             docstring += "\nRaises\n------\nException\n    Description of when this error is raised\n"
                            
#                         docstring += '"""'
                    
#                     elif style.lower() == "sphinx":
#                         docstring = f'"""Function to {name.replace("_", " ")}.

# This function {name.replace("_", " ")}.

# '
#                         for arg_name, arg_type in args:
#                             if arg_name != "self" and arg_name != "cls":
#                                 docstring += f":param {arg_name}: Description of {arg_name}\n"
#                                 if arg_type:
#                                     docstring += f":type {arg_name}: {arg_type}\n"
                        
#                         if returns:
#                             docstring += f":return: Description of return value\n"
#                             if return_type:
#                                 docstring += f":rtype: {return_type}\n"
                                
#                         if any('raise' in ast.unparse(expr) for expr in ast.walk(node) if isinstance(expr, ast.Expr)):
#                             docstring += ":raises: Exception: Description of when this error is raised\n"
                            
#                         docstring += '"""'
                    
#                 else:  # ClassDef
#                     # Extract class methods
#                     methods = [child.name for child in node.body if isinstance(child, ast.FunctionDef)]
                    
#                     # Extract class attributes (simplified)
#                     attributes = []
#                     for child in node.body:
#                         if isinstance(child, ast.Assign):
#                             for target in child.targets:
#                                 if isinstance(target, ast.Name):
#                                     attributes.append(target.id)
                    
#                     # Generate class docstring
#                     if style.lower() == "google":
#                         docstring = f'"""Class to {name.replace("_", " ")}.

# This class {name.replace("_", " ")}.

# Attributes:
# '
#                         for attr in attributes:
#                             docstring += f"    {attr}: Description of {attr}\n"
                        
#                         if methods:
#                             docstring += "\nMethods:\n"
#                             for method in methods:
#                                 if not method.startswith('__'):
#                                     docstring += f"    {method}: Description of {method}\n"
                                    
#                         docstring += '"""'
#                     elif style.lower() == "numpy":
#                         docstring = f'"""Class to {name.replace("_", " ")}.

# This class {name.replace("_", " ")}.

# Attributes
# ----------
# '
#                         for attr in attributes:
#                             docstring += f"{attr} : type\n    Description of {attr}\n"
                            
#                         if methods:
#                             docstring += "\nMethods\n-------\n"
#                             for method in methods:
#                                 if not method.startswith('__'):
#                                     docstring += f"{method}\n    Description of {method}\n"
                                    
#                         docstring += '"""'
#                     elif style.lower() == "sphinx":
#                         docstring = f'"""Class to {name.replace("_", " ")}.

# This class {name.replace("_", " ")}.

# '
#                         for attr in attributes:
#                             docstring += f":ivar {attr}: Description of {attr}\n"
                            
#                         if methods:
#                             for method in methods:
#                                 if not method.startswith('__'):
#                                     docstring += f":method {method}: Description of {method}\n"
                                    
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
                
#                 # Try to infer function behavior and parameters
#                 return_type = None
#                 if node.returns:
#                     try:
#                         return_type = ast.unparse(node.returns)
#                     except:
#                         pass
                
#                 # Try to guess argument types from defaults and annotations
#                 arg_types = {}
#                 for i, arg in enumerate(node.args.args):
#                     if arg.arg != 'self' and arg.arg != 'cls':
#                         # Check annotation
#                         if arg.annotation:
#                             try:
#                                 arg_types[arg.arg] = ast.unparse(arg.annotation)
#                             except:
#                                 pass
#                         # Check default value
#                         elif i < len(node.args.defaults):
#                             default_idx = i - (len(node.args.args) - len(node.args.defaults))
#                             if default_idx >= 0:
#                                 default_value = node.args.defaults[default_idx]
#                                 if isinstance(default_value, ast.Constant):
#                                     arg_types[arg.arg] = type(default_value.value).__name__
                
#                 # Generate appropriate test values based on types
#                 test_values = {}
#                 for arg in args:
#                     if arg in arg_types:
#                         arg_type = arg_types[arg]
#                         if 'int' in arg_type.lower():
#                             test_values[arg] = '42'
#                         elif 'float' in arg_type.lower():
#                             test_values[arg] = '3.14'
#                         elif 'str' in arg_type.lower() or 'string' in arg_type.lower():
#                             test_values[arg] = '"test_string"'
#                         elif 'bool' in arg_type.lower():
#                             test_values[arg] = 'True'
#                         elif 'list' in arg_type.lower():
#                             test_values[arg] = '[1, 2, 3]'
#                         elif 'dict' in arg_type.lower():
#                             test_values[arg] = '{"key": "value"}'
#                         else:
#                             test_values[arg] = 'None'
#                     else:
#                         test_values[arg] = 'None'
                
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
#     {chr(10).join([f'{arg} = {test_values[arg]}' for arg in args])}
    
#     # Execute
#     result = {function_name}({', '.join([arg for arg in args]) if args else ''})
    
#     # Assert
#     assert result is not None
#     # Add more specific assertions based on expected behavior


# def test_{function_name}_edge_cases():
#     """Test edge cases for {function_name}."""
#     # Edge case 1: Empty/zero values
#     {chr(10).join([f'{arg} = {empty_value_for_type(test_values[arg])}' for arg in args])}
#     result = {function_name}({', '.join([arg for arg in args]) if args else ''})
#     assert result is not None  # Replace with appropriate assertion
    
#     # Edge case 2: Large values
#     # TODO: Add test with large input values
    
#     # Edge case 3: Invalid input
#     # TODO: Add test with invalid input that should raise exception


# @pytest.mark.parametrize("input_values,expected", [
#     # TODO: Replace with actual test cases
#     ({{{', '.join([f'"{arg}": {test_values[arg]}' for arg in args])}}}, None),
#     # Add more test cases here
# ])
# def test_{function_name}_parametrized(input_values, expected):
#     """Parametrized test for different inputs."""
#     result = {function_name}(**input_values)
#     assert result == expected
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
#         {chr(10).join(['        ' + f'{arg} = {test_values[arg]}' for arg in args])}
#         result = {function_name}({', '.join([arg for arg in args]) if args else ''})
#         self.assertIsNotNone(result)
#         # Add more specific assertions based on expected behavior
    
#     def test_edge_cases(self):
#         # Edge case 1: Empty/zero values
#         {chr(10).join(['        ' + f'{arg} = {empty_value_for_type(test_values[arg])}' for arg in args])}
#         result = {function_name}({', '.join([arg for arg in args]) if args else ''})
#         self.assertIsNotNone(result)  # Replace with appropriate assertion
        
#         # Edge case 2: Large values
#         # TODO: Add test with large input values
        
#         # Edge case 3: Invalid input
#         # TODO: Add test to verify exception handling
    
#     def test_multiple_cases(self):
#         # Test multiple input combinations
#         test_cases = [
#             # (inputs, expected_output)
#             ({{{', '.join([f'{arg}: {test_values[arg]}' for arg in args])}}}, None),
#             # Add more test cases here
#         ]
        
#         for inputs, expected in test_cases:
#             with self.subTest(inputs=inputs):
#                 result = {function_name}(**inputs)
#                 self.assertEqual(result, expected)


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

# def empty_value_for_type(value_str: str) -> str:
#     """Helper function to generate empty/minimal values for a given type."""
#     if value_str.startswith('"') or value_str.startswith("'"):
#         return '""'  # Empty string
#     elif value_str.startswith('['):
#         return '[]'  # Empty list
#     elif value_str.startswith('{'):
#         return '{}'  # Empty dict
#     elif value_str == 'True' or value_str == 'False':
#         return 'False'  # False as minimal boolean
#     elif '.' in value_str and value_str.replace('.', '').isdigit():
#         return '0.0'  # Zero float
#     elif value_str.isdigit():
#         return '0'  # Zero int
#     else:
#         return 'None'  # Default to None

# @register_tool(
#     name="refactor_code",
#     description="Refactor code to improve structure and readability."
# )
# def refactor_code(code: str, refactoring_type: str = "general") -> str:
#     """Refactor code to improve structure and readability.
    
#     Args:
#         code: The code to refactor
#         refactoring_type: The type of refactoring to perform
#             (general, extract_method, rename_variables, etc.)
        
#     Returns:
#         The refactored code
#     """
#     try:
#         logger.info(f"Refactoring code with approach: {refactoring_type}")
        
#         # Parse the code
#         try:
#             tree = ast.parse(code)
#         except SyntaxError as e:
#             return f"# Error parsing code: {str(e)}\n" + code
        
#         # Map from refactoring type to refactoring function
#         refactoring_functions = {
#             "general": lambda c: general_refactoring(c),
#             "extract_method": lambda c: extract_method_refactoring(c),
#             "rename_variables": lambda c: rename_variables_refactoring(c),
#             "simplify_conditionals": lambda c: simplify_conditionals_refactoring(c),
#             "optimize_imports": lambda c: optimize_imports_refactoring(c),
#         }
        
#         # If the refactoring type is not supported, do general refactoring
#         refactoring_func = refactoring_functions.get(refactoring_type.lower(), refactoring_functions["general"])
        
#         # Perform the refactoring
#         return refactoring_func(code)
    
#     except Exception as e:
#         logger.error(f"Error refactoring code: {str(e)}")
#         return f"# Error refactoring code: {str(e)}\n" + code

# def general_refactoring(code: str) -> str:
#     """Perform general refactoring improvements."""
#     try:
#         # Parse the code
#         tree = ast.parse(code)
        
#         # Format with black if available
#         if HAS_CODE_DEPS:
#             formatted_code = black.format_str(code, mode=black.Mode())
#         else:
#             formatted_code = code
        
#         # Identify long functions (>50 lines)
#         long_functions = []
#         for node in ast.walk(tree):
#             if isinstance(node, ast.FunctionDef):
#                 func_code = ast.unparse(node)
#                 if len(func_code.split("\n")) > 50:
#                     long_functions.append(node.name)
        
#         # Add comments for potential refactoring
#         if long_functions:
#             formatted_code = "# Refactoring suggestions:\n"
#             for func_name in long_functions:
#                 formatted_code += f"# - Consider breaking '{func_name}' into smaller functions\n"
#             formatted_code += "\n" + (formatted_code if formatted_code else code)
        
#         # Add a flag for unused imports
#         import_names = set()
#         for node in ast.walk(tree):
#             if isinstance(node, ast.Import):
#                 for name in node.names:
#                     import_names.add(name.name)
#             elif isinstance(node, ast.ImportFrom):
#                 module_name = node.module or ""
#                 for name in node.names:
#                     if name.name != "*":
#                         import_names.add(f"{module_name}.{name.name}")
        
#         # Check for used imports (simplified)
#         used_names = set()
#         for node in ast.walk(tree):
#             if isinstance(node, ast.Name):
#                 used_names.add(node.id)
#             elif isinstance(node, ast.Attribute):
#                 if isinstance(node.value, ast.Name):
#                     used_names.add(f"{node.value.id}.{node.attr}")
        
#         # Find potentially unused imports
#         unused_imports = import_names - used_names
#         if unused_imports:
#             formatted_code = "# Possible unused imports:\n"
#             for imp in unused_imports:
#                 formatted_code += f"# - {imp}\n"
#             formatted_code += "\n" + (formatted_code if formatted_code else code)
        
#         return formatted_code
    
#     except Exception as e:
#         logger.error(f"Error in general refactoring: {str(e)}")
#         return code

# def extract_method_refactoring(code: str) -> str:
#     """Refactor by suggesting extract method opportunities."""
#     try:
#         # Parse the code
#         tree = ast.parse(code)
        
#         # Look for complex code blocks that could be extracted
#         suggestions = []
        
#         for node in ast.walk(tree):
#             if isinstance(node, ast.FunctionDef):
#                 block_count = 0
#                 complex_blocks = []
                
#                 # Count blocks of code (if/for/while) inside the function
#                 for child in ast.walk(node):
#                     if isinstance(child, (ast.If, ast.For, ast.While)):
#                         block_count += 1
                        
#                         # Check if the block is complex enough to extract
#                         block_str = ast.unparse(child)
#                         if len(block_str.split("\n")) > 10:
#                             complex_blocks.append(child)
                
#                 # If the function has complex blocks, suggest extraction
#                 if complex_blocks:
#                     for i, block in enumerate(complex_blocks):
#                         block_type = type(block).__name__
#                         suggestions.append(f"In function '{node.name}', consider extracting the {block_type} block at line {block.lineno} into a separate function")
        
#         # Format the code
#         if HAS_CODE_DEPS:
#             formatted_code = black.format_str(code, mode=black.Mode())
#         else:
#             formatted_code = code
            
#         # Add extraction suggestions as comments
#         if suggestions:
#             result = "# Extract Method Suggestions:\n"
#             for suggestion in suggestions:
#                 result += f"# - {suggestion}\n"
#             result += "\n" + formatted_code
#             return result
        
#         return formatted_code
    
#     except Exception as e:
#         logger.error(f"Error in extract method refactoring: {str(e)}")
#         return code

# def rename_variables_refactoring(code: str) -> str:
#     """Refactor by suggesting better variable names."""
#     try:
#         # Parse the code
#         tree = ast.parse(code)
        
#         # Look for single-character variable names
#         single_char_vars = set()
#         loop_vars = set()
        
#         for node in ast.walk(tree):
#             # Find variable assignments
#             if isinstance(node, ast.Assign):
#                 for target in node.targets:
#                     if isinstance(target, ast.Name):
#                         if len(target.id) == 1 and target.id not in ['i', 'j', 'k', 'x', 'y', 'z']:
#                             single_char_vars.add(target.id)
            
#             # Find loop variables
#             elif isinstance(node, ast.For):
#                 if isinstance(node.target, ast.Name):
#                     loop_vars.add(node.target.id)
        
#         # Remove loop variables from single_char_vars
#         single_char_vars -= loop_vars
        
#         # Format the code
#         if HAS_CODE_DEPS:
#             formatted_code = black.format_str(code, mode=black.Mode())
#         else:
#             formatted_code = code
            
#         # Add variable renaming suggestions
#         if single_char_vars:
#             result = "# Variable Naming Suggestions:\n"
#             for var in single_char_vars:
#                 result += f"# - Consider renaming variable '{var}' to something more descriptive\n"
#             result += "\n" + formatted_code
#             return result
        
#         return formatted_code
    
#     except Exception as e:
#         logger.error(f"Error in rename variables refactoring: {str(e)}")
#         return code

# def simplify_conditionals_refactoring(code: str) -> str:
#     """Refactor by simplifying complex conditional expressions."""
#     try:
#         # Parse the code
#         tree = ast.parse(code)
        
#         # Look for complex conditional expressions
#         complex_conditionals = []
        
#         for node in ast.walk(tree):
#             if isinstance(node, ast.If):
#                 # Check for nested conditions
#                 nested_conditions = 0
#                 for child in ast.walk(node.test):
#                     if isinstance(child, (ast.BoolOp, ast.Compare)):
#                         nested_conditions += 1
                
#                 if nested_conditions > 2:
#                     complex_conditionals.append((node.lineno, ast.unparse(node.test)))
        
#         # Format the code
#         if HAS_CODE_DEPS:
#             formatted_code = black.format_str(code, mode=black.Mode())
#         else:
#             formatted_code = code
            
#         # Add conditional simplification suggestions
#         if complex_conditionals:
#             result = "# Conditional Simplification Suggestions:\n"
#             for line, cond in complex_conditionals:
#                 result += f"# - Consider simplifying complex condition at line {line}:\n#   {cond}\n"
#             result += "\n" + formatted_code
#             return result
        
#         return formatted_code
    
#     except Exception as e:
#         logger.error(f"Error in simplify conditionals refactoring: {str(e)}")
#         return code

# def optimize_imports_refactoring(code: str) -> str:
#     """Refactor by organizing and optimizing imports."""
#     try:
#         # If isort is available, use it to organize imports
#         if HAS_CODE_DEPS:
#             try:
#                 formatted_code = isort.code(code)
#                 return formatted_code
#             except Exception as e:
#                 logger.warning(f"Error using isort: {str(e)}")
        
#         # Manual import organization if isort is not available
#         try:
#             tree = ast.parse(code)
            
#             # Collect all import statements
#             imports = []
#             import_nodes = []
#             for i, node in enumerate(tree.body):
#                 if isinstance(node, (ast.Import, ast.ImportFrom)):
#                     import_str = ast.unparse(node)
#                     imports.append(import_str)
#                     import_nodes.append((i, node))
            
#             # Skip if no imports found
#             if not imports:
#                 return code
            
#             # Sort imports by type and name
#             std_imports = []
#             third_party_imports = []
#             local_imports = []
            
#             for imp in imports:
#                 if imp.startswith("import "):
#                     module = imp.replace("import ", "").strip()
#                     if module in sys.builtin_module_names or module == "sys":
#                         std_imports.append(imp)
#                     else:
#                         third_party_imports.append(imp)
#                 elif imp.startswith("from "):
#                     module = imp.split(" import ")[0].replace("from ", "").strip()
#                     if "." in module or module in sys.builtin_module_names:
#                         std_imports.append(imp)
#                     else:
#                         local_imports.append(imp)
            
#             # Sort each group
#             std_imports.sort()
#             third_party_imports.sort()
#             local_imports.sort()
            
#             # Combine sorted imports
#             organized_imports = std_imports + ["\n"] + third_party_imports + ["\n"] + local_imports
            
#             # Replace imports in the original code
#             lines = code.split("\n")
#             import_lines = []
#             for i, node in sorted(import_nodes, reverse=True):
#                 start_line = node.lineno - 1
#                 end_line = start_line
#                 if isinstance(node, ast.Import):
#                     end_line += len(node.names) - 1
#                 elif isinstance(node, ast.ImportFrom):
#                     end_line += len(node.names) - 1
                
#                 # Remove the import lines
#                 del lines[start_line:end_line+1]
                
#             # Add organized imports at the top
#             organized_code = "\n".join(organized_imports) + "\n\n" + "\n".join(lines)
#             return organized_code
            
#         except Exception as e:
#             logger.warning(f"Error in manual import organization: {str(e)}")
#             return code
    
#     except Exception as e:
#         logger.error(f"Error in optimize imports refactoring: {str(e)}")
#         return code

# @register_tool(
#     name="generate_cli",
#     description="Generate a command-line interface for a function or module."
# )
# def generate_cli(code: str, cli_framework: str = "argparse") -> str:
#     """Generate a command-line interface for a function or module.
    
#     Args:
#         code: The function or module code
#         cli_framework: The CLI framework to use (argparse, click, etc.)
        
#     Returns:
#         The code with added CLI functionality
#     """
#     try:
#         logger.info(f"Generating CLI using {cli_framework}")
        
#         # Parse the code
#         try:
#             tree = ast.parse(code)
#         except SyntaxError as e:
#             return f"# Error parsing code: {str(e)}\n" + code
        
#         # Extract functions and their parameters
#         functions = []
#         for node in ast.walk(tree):
#             if isinstance(node, ast.FunctionDef):
#                 # Skip private functions
#                 if node.name.startswith('_'):
#                     continue
                
#                 # Get function parameters
#                 params = []
#                 for arg in node.args.args:
#                     if arg.arg not in ['self', 'cls']:
#                         param_type = None
#                         default_value = None
                        
#                         # Check for type annotation
#                         if arg.annotation:
#                             try:
#                                 param_type = ast.unparse(arg.annotation)
#                             except:
#                                 pass
                        
#                         # Check for default value
#                         arg_idx = node.args.args.index(arg)
#                         defaults_offset = len(node.args.args) - len(node.args.defaults)
#                         if arg_idx >= defaults_offset and node.args.defaults:
#                             default_idx = arg_idx - defaults_offset
#                             if default_idx < len(node.args.defaults):
#                                 try:
#                                     default_value = ast.unparse(node.args.defaults[default_idx])
#                                 except:
#                                     pass
                        
#                         params.append({
#                             'name': arg.arg,
#                             'type': param_type,
#                             'default': default_value
#                         })
                
#                 # Get docstring for help text
#                 docstring = ast.get_docstring(node) or f"Function to {node.name.replace('_', ' ')}"
#                 docstring_short = docstring.split('\n')[0] if docstring else ""
                
#                 functions.append({
#                     'name': node.name,
#                     'params': params,
#                     'docstring': docstring_short
#                 })
        
#         # Generate CLI code based on the framework
#         if cli_framework.lower() == "argparse":
#             return generate_argparse_cli(code, functions)
#         elif cli_framework.lower() == "click":
#             return generate_click_cli(code, functions)
#         else:
#             return f"# Unsupported CLI framework: {cli_framework}\n# Supported frameworks: argparse, click\n" + code
    
#     except Exception as e:
#         logger.error(f"Error generating CLI: {str(e)}")
#         return f"# Error generating CLI: {str(e)}\n" + code

# def generate_argparse_cli(code: str, functions: List[Dict]) -> str:
#     """Generate CLI code using argparse."""
#     if not functions:
#         return code + "\n\n# No suitable functions found for CLI generation\n"
    
#     # Start with the original code
#     result = code.rstrip() + "\n\n"
    
#     # Add imports
#     result += "\n# CLI interface\nimport argparse\nimport sys\n\n"
    
#     # Generate main function
#     result += "def main():\n"
#     result += "    parser = argparse.ArgumentParser(description='Command line interface')\n"
#     result += "    subparsers = parser.add_subparsers(dest='command', help='Commands')\n\n"
    
#     # Generate subparsers for each function
#     for func in functions:
#         result += f"    # Subparser for {func['name']}\n"
#         result += f"    {func['name']}_parser = subparsers.add_parser('{func['name'].replace('_', '-')}', help='{func['docstring']}')\n"
        
#         # Add arguments for each parameter
#         for param in func['params']:
#             param_name = param['name'].replace('_', '-')
#             arg_name = f"--{param_name}"
            
#             # Determine argument type
#             arg_type = "str"  # Default type
#             if param['type']:
#                 if "int" in param['type']:
#                     arg_type = "int"
#                 elif "float" in param['type']:
#                     arg_type = "float"
#                 elif "bool" in param['type']:
#                     arg_type = "bool"
#                 elif "list" in param['type'] or "List" in param['type']:
#                     arg_type = "list"
            
#             # Determine if required
#             required = "True" if param['default'] is None else "False"
            
#             # Handle default value
#             default_clause = ""
#             if param['default'] is not None:
#                 default_clause = f", default={param['default']}"
            
#             # Add the argument
#             if arg_type == "bool":
#                 result += f"    {func['name']}_parser.add_argument('{arg_name}', action='store_true', help='{param['name']}{default_clause}')\n"
#             elif arg_type == "list":
#                 result += f"    {func['name']}_parser.add_argument('{arg_name}', nargs='+', help='{param['name']}'{default_clause})\n"
#             else:
#                 result += f"    {func['name']}_parser.add_argument('{arg_name}', type={arg_type}, required={required}, help='{param['name']}'{default_clause})\n"
        
#         result += "\n"
    
#     # Handle arguments and call functions
#     result += "    args = parser.parse_args()\n\n"
#     result += "    if not args.command:\n"
#     result += "        parser.print_help()\n"
#     result += "        return 1\n\n"
    
#     # Call appropriate function based on command
#     result += "    # Convert args to kwargs and call the appropriate function\n"
#     result += "    kwargs = {k: v for k, v in vars(args).items() if k != 'command'}\n"
#     for func in functions:
#         result += f"    if args.command == '{func['name'].replace('_', '-')}':\n"
#         result += f"        result = {func['name']}(**kwargs)\n"
#         result += "        print(result)\n"
    
#     result += "    return 0\n\n"
    
#     # Add if __name__ == '__main__' block
#     result += "if __name__ == '__main__':\n"
#     result += "    sys.exit(main())\n"
    
#     return result

# def generate_click_cli(code: str, functions: List[Dict]) -> str:
#     """Generate CLI code using click."""
#     if not functions:
#         return code + "\n\n# No suitable functions found for CLI generation\n"
    
#     # Start with the original code
#     result = code.rstrip() + "\n\n"
    
#     # Add imports
#     result += "\n# CLI interface\nimport click\n\n"
    
#     # Generate main CLI group
#     result += "@click.group()\n"
#     result += "def cli():\n"
#     result += "    \"\"\"Command line interface.\"\"\"\n"
#     result += "    pass\n\n"
    
#     # Generate commands for each function
#     for func in functions:
#         result += f"@cli.command('{func['name'].replace('_', '-')}')\n"
        
#         # Add options for each parameter
#         for param in func['params']:
#             param_name = param['name']
#             option_name = f"--{param_name.replace('_', '-')}"
            
#             # Determine parameter type
#             param_type = "str"  # Default type
#             if param['type']:
#                 if "int" in param['type']:
#                     param_type = "int"
#                 elif "float" in param['type']:
#                     param_type = "float"
#                 elif "bool" in param['type']:
#                     param_type = "bool"
#                 elif "list" in param['type'] or "List" in param['type']:
#                     param_type = "list"
            
#             # Handle default value
#             default_clause = ""
#             if param['default'] is not None:
#                 default_clause = f", default={param['default']}"
            
#             # Add the option
#             if param_type == "bool":
#                 result += f"@click.option('{option_name}', is_flag=True, help='{param_name}'{default_clause})\n"
#             elif param_type == "list":
#                 result += f"@click.option('{option_name}', multiple=True, help='{param_name}'{default_clause})\n"
#             else:
#                 result += f"@click.option('{option_name}', type={param_type.capitalize()}, help='{param_name}'{default_clause})\n"
        
#         # Function body
#         result += f"def {func['name']}_cmd({', '.join([p['name'] for p in func['params']])}):\n"
#         result += f"    \"\"\"${func['docstring']}\"\"\"\n"
#         result += f"    result = {func['name']}({', '.join([p['name'] + '=' + p['name'] for p in func['params']])})\n"
#         result += "    click.echo(result)\n\n"
    
#     # Add if __name__ == '__main__' block
#     result += "if __name__ == '__main__':\n"
#     result += "    cli()\n"
    
#     return result

# @register_tool(
#     name="identify_code_vulnerabilities",
#     description="Identify potential security vulnerabilities in code."
# )
# def identify_code_vulnerabilities(code: str, language: str = "python") -> List[Dict[str, str]]:
#     """Identify potential security vulnerabilities in code.
    
#     Args:
#         code: The code to analyze
#         language: The programming language of the code
        
#     Returns:
#         A list of identified vulnerabilities
#     """
#     try:
#         logger.info(f"Analyzing {language} code for vulnerabilities")
        
#         vulnerabilities = []
        
#         if language.lower() == "python":
#             try:
#                 # Parse the code
#                 tree = ast.parse(code)
                
#                 # Check for OS command injection
#                 for node in ast.walk(tree):
#                     # Check for os.system, os.popen, subprocess calls with user input
#                     if isinstance(node, ast.Call):
#                         # Check if it's a subprocess or os call
#                         func_name = ""
#                         if isinstance(node.func, ast.Name):
#                             func_name = node.func.id
#                         elif isinstance(node.func, ast.Attribute):
#                             if isinstance(node.func.value, ast.Name):
#                                 if node.func.value.id in ['os', 'subprocess']:
#                                     func_name = f"{node.func.value.id}.{node.func.attr}"
                        
#                         # Check for dangerous functions
#                         dangerous_funcs = [
#                             'os.system', 'os.popen', 'os.spawn', 'os.exec',
#                             'subprocess.run', 'subprocess.call', 'subprocess.Popen',
#                             'eval', 'exec'
#                         ]
                        
#                         if func_name in dangerous_funcs or func_name.startswith(tuple(dangerous_funcs)):
#                             vulnerabilities.append({
#                                 'type': 'Command Injection',
#                                 'severity': 'High',
#                                 'line': node.lineno,
#                                 'description': f"Potential command injection in {func_name} call",
#                                 'recommendation': "Use safe APIs or input validation to prevent injection attacks"
#                             })
                        
#                         # Check for SQL injection
#                         if isinstance(node.func, ast.Attribute):
#                             if node.func.attr in ['execute', 'executemany', 'executescript']:
#                                 vulnerabilities.append({
#                                     'type': 'SQL Injection',
#                                     'severity': 'High',
#                                     'line': node.lineno,
#                                     'description': f"Potential SQL injection in {node.func.attr} call",
#                                     'recommendation': "Use parameterized queries or prepared statements"
#                                 })
                
#                 # Check for file operations without validation
#                 for node in ast.walk(tree):
#                     if isinstance(node, ast.Call):
#                         if isinstance(node.func, ast.Name) and node.func.id in ['open']:
#                             vulnerabilities.append({
#                                 'type': 'Unsafe File Operation',
#                                 'severity': 'Medium',
#                                 'line': node.lineno,
#                                 'description': "File operation without path validation",
#                                 'recommendation': "Validate and sanitize file paths before operations"
#                             })
                
#                 # Check for pickle, yaml, and marshal usage
#                 for node in ast.walk(tree):
#                     if isinstance(node, (ast.Import, ast.ImportFrom)):
#                         for name in node.names:
#                             if name.name in ['pickle', 'yaml', 'marshal']:
#                                 vulnerabilities.append({
#                                     'type': 'Unsafe Deserialization',
#                                     'severity': 'High',
#                                     'line': node.lineno,
#                                     'description': f"Usage of potentially unsafe {name.name} module",
#                                     'recommendation': "Avoid deserializing untrusted data"
#                                 })
                
#                 # Check for hardcoded credentials (simplified)
#                 for node in ast.walk(tree):
#                     if isinstance(node, ast.Assign):
#                         for target in node.targets:
#                             if isinstance(target, ast.Name):
#                                 var_name = target.id.lower()
#                                 sensitive_names = ['password', 'secret', 'key', 'token', 'api_key', 'auth']
#                                 if any(s in var_name for s in sensitive_names):
#                                     if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
#                                         vulnerabilities.append({
#                                             'type': 'Hardcoded Secret',
#                                             'severity': 'High',
#                                             'line': node.lineno,
#                                             'description': f"Hardcoded credential in variable '{target.id}'",
#                                             'recommendation': "Use environment variables or a secure configuration manager"
#                                         })
            
#             except SyntaxError as e:
#                 vulnerabilities.append({
#                     'type': 'Syntax Error',
#                     'severity': 'Low',
#                     'line': e.lineno,
#                     'description': f"Syntax error: {str(e)}",
#                     'recommendation': "Fix the syntax error"
#                 })
        
#         elif language.lower() in ["javascript", "typescript", "js", "ts"]:
#             # Simple pattern-based analysis for JS/TS
            
#             # Check for eval use
#             eval_matches = re.finditer(r'\beval\s*\(', code)
#             for match in eval_matches:
#                 line_no = code[:match.start()].count('\n') + 1
#                 vulnerabilities.append({
#                     'type': 'Code Injection',
#                     'severity': 'High',
#                     'line': line_no,
#                     'description': "Use of eval() can lead to code injection",
#                     'recommendation': "Avoid using eval with untrusted input"
#                 })
            
#             # Check for innerHTML assignments
#             inner_html_matches = re.finditer(r'\.innerHTML\s*=', code)
#             for match in inner_html_matches:
#                 line_no = code[:match.start()].count('\n') + 1
#                 vulnerabilities.append({
#                     'type': 'Cross-Site Scripting (XSS)',
#                     'severity': 'High',
#                     'line': line_no,
#                     'description': "Direct innerHTML assignment can lead to XSS",
#                     'recommendation': "Use textContent or DOM methods instead"
#                 })
            
#             # Check for potentially unsafe URL handling
#             url_matches = re.finditer(r'location\.(href|replace|assign)\s*=', code)
#             for match in url_matches:
#                 line_no = code[:match.start()].count('\n') + 1
#                 vulnerabilities.append({
#                     'type': 'Open Redirect',
#                     'severity': 'Medium',
#                     'line': line_no,
#                     'description': "Potential open redirect vulnerability",
#                     'recommendation': "Validate URLs before redirection"
#                 })
            
#             # Check for hardcoded secrets
#             secret_matches = re.finditer(r'(password|secret|key|token|auth)\s*[=:]\s*["\'][^"\']+["\']', code, re.IGNORECASE)
#             for match in secret_matches:
#                 line_no = code[:match.start()].count('\n') + 1
#                 vulnerabilities.append({
#                     'type': 'Hardcoded Secret',
#                     'severity': 'High',
#                     'line': line_no,
#                     'description': "Hardcoded credential detected",
#                     'recommendation': "Use environment variables or a secure configuration manager"
#                 })
        
#         return vulnerabilities
    
#     except Exception as e:
#         logger.error(f"Error identifying vulnerabilities: {str(e)}")
#         return [{'type': 'Error', 'severity': 'Unknown', 'line': 0, 'description': f"Error during analysis: {str(e)}", 'recommendation': "Fix the error"}]