"""Data processing tools for Agnassan.

This module provides tools for data manipulation, analysis, and visualization
that enhance the capabilities of language models.
"""

import logging
import json
import csv
import io
from typing import Dict, List, Any, Optional, Union
import re

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_DATA_DEPS = True
except ImportError:
    HAS_DATA_DEPS = False

from .index import register_tool

# Set up logging
logger = logging.getLogger("agnassan.tools.data")

@register_tool(
    name="parse_json",
    description="Parse a JSON string into a Python object."
)
def parse_json(json_str: str) -> Any:
    """Parse a JSON string into a Python object.
    
    Args:
        json_str: The JSON string to parse
        
    Returns:
        The parsed Python object (dict, list, etc.)
    """
    try:
        logger.info("Parsing JSON string")
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        return {"error": str(e)}

@register_tool(
    name="to_json",
    description="Convert a Python object to a JSON string."
)
def to_json(obj: Any, pretty: bool = False) -> str:
    """Convert a Python object to a JSON string.
    
    Args:
        obj: The Python object to convert
        pretty: Whether to format the JSON with indentation
        
    Returns:
        The JSON string representation
    """
    try:
        logger.info("Converting object to JSON")
        indent = 2 if pretty else None
        return json.dumps(obj, indent=indent)
    except Exception as e:
        logger.error(f"Error converting to JSON: {str(e)}")
        return f"{{\"error\": \"{str(e)}\"}}"

@register_tool(
    name="parse_csv",
    description="Parse a CSV string into a list of dictionaries."
)
def parse_csv(csv_str: str, delimiter: str = ',') -> List[Dict[str, str]]:
    """Parse a CSV string into a list of dictionaries.
    
    Args:
        csv_str: The CSV string to parse
        delimiter: The delimiter character (default: ',')
        
    Returns:
        A list of dictionaries, where each dictionary represents a row
    """
    try:
        logger.info(f"Parsing CSV string with delimiter '{delimiter}'")
        reader = csv.DictReader(io.StringIO(csv_str), delimiter=delimiter)
        return list(reader)
    except Exception as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        return [{"error": str(e)}]

@register_tool(
    name="to_csv",
    description="Convert a list of dictionaries to a CSV string."
)
def to_csv(data: List[Dict[str, Any]], delimiter: str = ',') -> str:
    """Convert a list of dictionaries to a CSV string.
    
    Args:
        data: A list of dictionaries, where each dictionary represents a row
        delimiter: The delimiter character (default: ',')
        
    Returns:
        The CSV string representation
    """
    try:
        logger.info(f"Converting data to CSV with delimiter '{delimiter}'")
        if not data:
            return ""
        
        output = io.StringIO()
        fieldnames = data[0].keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error converting to CSV: {str(e)}")
        return f"error,message\n,{str(e)}"

@register_tool(
    name="analyze_data",
    description="Analyze a dataset and return basic statistics."
)
def analyze_data(data: Union[List[Dict[str, Any]], str]) -> Dict[str, Any]:
    """Analyze a dataset and return basic statistics.
    
    Args:
        data: Either a list of dictionaries or a CSV/JSON string
        
    Returns:
        A dictionary containing basic statistics about the data
    """
    try:
        logger.info("Analyzing data")
        
        # Check if pandas is available
        if not HAS_DATA_DEPS:
            return {"error": "Pandas is not available. Install pandas for data analysis."}
        
        # Convert string to data if needed
        if isinstance(data, str):
            if data.strip().startswith('{') or data.strip().startswith('['):
                # Assume JSON
                data = json.loads(data)
            else:
                # Assume CSV
                data = parse_csv(data)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Basic statistics
        result = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "summary": {}
        }
        
        # Summary statistics for numeric columns
        for col in df.select_dtypes(include=['number']).columns:
            result["summary"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std())
            }
        
        # Value counts for categorical columns (limited to top 5)
        for col in df.select_dtypes(exclude=['number']).columns:
            result["summary"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return {"error": str(e)}

@register_tool(
    name="filter_data",
    description="Filter a dataset based on a condition."
)
def filter_data(data: List[Dict[str, Any]], condition: str) -> List[Dict[str, Any]]:
    """Filter a dataset based on a condition.
    
    Args:
        data: A list of dictionaries, where each dictionary represents a row
        condition: A string representing a Python expression that evaluates to a boolean
                  The expression can use column names as variables
        
    Returns:
        A filtered list of dictionaries
    """
    try:
        logger.info(f"Filtering data with condition: {condition}")
        
        # Simple condition parser
        # This is a basic implementation and has security implications in a real system
        # A more robust solution would use a proper query language or parser
        result = []
        for row in data:
            # Create a local context with the row data
            context = row.copy()
            # Evaluate the condition in the context
            if eval(condition, {"__builtins__": {}}, context):
                result.append(row)
        
        return result
    except Exception as e:
        logger.error(f"Error filtering data: {str(e)}")
        return [{"error": str(e)}]

@register_tool(
    name="generate_chart_data",
    description="Generate data for a chart based on the provided dataset."
)
def generate_chart_data(data: List[Dict[str, Any]], chart_type: str, 
                       x_column: str, y_column: str = None, 
                       group_by: str = None) -> Dict[str, Any]:
    """Generate data for a chart based on the provided dataset.
    
    Args:
        data: A list of dictionaries, where each dictionary represents a row
        chart_type: The type of chart ('bar', 'line', 'pie', 'scatter')
        x_column: The column to use for the x-axis
        y_column: The column to use for the y-axis (not needed for pie charts)
        group_by: Optional column to group the data by
        
    Returns:
        A dictionary containing the chart data in a format suitable for visualization
    """
    try:
        logger.info(f"Generating {chart_type} chart data")
        
        # Check if pandas is available
        if not HAS_DATA_DEPS:
            return {"error": "Pandas is not available. Install pandas for chart generation."}
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        result = {
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "group_by": group_by,
            "data": {}
        }
        
        if chart_type == 'pie':
            # For pie charts, we need counts or sums
            if y_column:
                # Sum values by category
                series = df.groupby(x_column)[y_column].sum()
            else:
                # Count occurrences of each category
                series = df[x_column].value_counts()
            
            result["data"] = {
                "labels": series.index.tolist(),
                "values": series.values.tolist()
            }
            
        elif chart_type in ['bar', 'line']:
            if group_by:
                # Grouped chart
                groups = df.groupby([x_column, group_by])
                if y_column:
                    # Aggregate values
                    agg_data = groups[y_column].agg(['mean', 'sum', 'count']).reset_index()
                    pivot = agg_data.pivot(index=x_column, columns=group_by, values='sum')
                else:
                    # Count occurrences
                    counts = groups.size().reset_index(name='count')
                    pivot = counts.pivot(index=x_column, columns=group_by, values='count')
                
                result["data"] = {
                    "x": pivot.index.tolist(),
                    "series": [
                        {
                            "name": str(col),
                            "values": pivot[col].fillna(0).tolist()
                        } for col in pivot.columns
                    ]
                }
            else:
                # Simple chart
                if y_column:
                    # Aggregate values
                    agg_data = df.groupby(x_column)[y_column].agg(['mean', 'sum', 'count']).reset_index()
                    result["data"] = {
                        "x": agg_data[x_column].tolist(),
                        "y": agg_data['sum'].tolist()
                    }
                else:
                    # Count occurrences
                    counts = df[x_column].value_counts().reset_index()
                    counts.columns = [x_column, 'count']
                    result["data"] = {
                        "x": counts[x_column].tolist(),
                        "y": counts['count'].tolist()
                    }
                    
        elif chart_type == 'scatter':
            if not y_column:
                return {"error": "y_column is required for scatter plots"}
            
            if group_by:
                # Grouped scatter plot
                result["data"] = {
                    "series": []
                }
                for group_val, group_df in df.groupby(group_by):
                    result["data"]["series"].append({
                        "name": str(group_val),
                        "x": group_df[x_column].tolist(),
                        "y": group_df[y_column].tolist()
                    })
            else:
                # Simple scatter plot
                result["data"] = {
                    "x": df[x_column].tolist(),
                    "y": df[y_column].tolist()
                }
        
        return result
    except Exception as e:
        logger.error(f"Error generating chart data: {str(e)}")
        return {"error": str(e)}