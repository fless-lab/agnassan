"""Web tools for Agnassan.

This module provides tools for web scraping, URL fetching, and other web-related
functionalities that enhance the capabilities of language models.
"""

import logging
import requests
from typing import Dict, List, Optional, Union
from bs4 import BeautifulSoup
import json
import re

from .index import register_tool

# Set up logging
logger = logging.getLogger("agnassan.tools.web")

@register_tool(
    name="web_search",
    description="Search the web for information on a given query using DuckDuckGo."
)
def web_search(query: str, num_results: int = 5, region: str = "wt-wt", safesearch: str = "moderate") -> List[Dict[str, str]]:
    """Search the web for information on a given query using DuckDuckGo.
    
    This function uses DuckDuckGo to find relevant information on the web.
    It returns a list of search results with titles, snippets, and URLs.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5)
        region: Region for search results (default: "wt-wt" for worldwide)
        safesearch: SafeSearch setting ("off", "moderate", or "strict", default: "moderate")
        
    Returns:
        A list of dictionaries containing search results with keys:
        - title: The title of the search result
        - snippet: A short excerpt from the page
        - url: The URL of the page
    """
    try:
        logger.info(f"Searching web for: {query} using DuckDuckGo")
        
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("DuckDuckGo search package not installed. Install with 'pip install duckduckgo-search'")
            return [{"title": "Error", "snippet": "DuckDuckGo search package not installed. Install with 'pip install duckduckgo-search'", "url": ""}]
        
        # Initialize DuckDuckGo search
        ddgs = DDGS()
        
        # Perform the search
        results = []
        for r in ddgs.text(query, region=region, safesearch=safesearch, max_results=num_results):
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", "")
            })
        
        return results
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        return [{"title": "Error", "snippet": f"Failed to search: {str(e)}", "url": ""}]

@register_tool(
    name="fetch_url",
    description="Fetch the content of a URL and return the HTML or text content."
)
def fetch_url(url: str, as_text: bool = True) -> str:
    """Fetch the content of a URL.
    
    Args:
        url: The URL to fetch
        as_text: Whether to return the content as text (True) or HTML (False)
        
    Returns:
        The content of the URL as text or HTML
    """
    try:
        logger.info(f"Fetching URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        if as_text:
            # Parse HTML and extract text
            soup = BeautifulSoup(response.content, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            # Get text
            text = soup.get_text(separator="\n", strip=True)
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            return text
        else:
            return response.text
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return f"Error: {str(e)}"

@register_tool(
    name="extract_from_html",
    description="Extract specific information from HTML content using CSS selectors."
)
def extract_from_html(html: str, selector: str) -> List[str]:
    """Extract specific information from HTML content using CSS selectors.
    
    Args:
        html: The HTML content to parse
        selector: CSS selector to extract elements
        
    Returns:
        A list of extracted text from matching elements
    """
    try:
        logger.info(f"Extracting from HTML using selector: {selector}")
        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector)
        return [element.get_text(strip=True) for element in elements]
    except Exception as e:
        logger.error(f"Error extracting from HTML: {str(e)}")
        return [f"Error: {str(e)}"]

@register_tool(
    name="summarize_webpage",
    description="Fetch a webpage and summarize its main content using NLP tools."
)
def summarize_webpage(url: str, sentences: int = 3) -> str:
    """Fetch a webpage and summarize its main content using NLP tools.
    
    This function fetches a webpage, extracts its main content, and returns a summary
    using the NLP summarization tool.
    
    Args:
        url: The URL of the webpage to summarize
        sentences: Number of sentences to include in the summary (default: 3)
        
    Returns:
        A summary of the webpage content
    """
    try:
        # Fetch the webpage content
        content = fetch_url(url)
        
        # Try to use the NLP summarize_text tool if available
        try:
            from agnassan.tools.nlp_tools import summarize_text
            return summarize_text(content, sentences)
        except (ImportError, AttributeError):
            # Fall back to simple summarization if NLP tools aren't available
            logger.warning("NLP summarize_text tool not available, using simple summarization")
            
            # Simple extractive summarization approach
            sentences_list = re.split(r'(?<=[.!?])\s+', content)
            
            # Return the full text if it's already shorter than requested summary
            if len(sentences_list) <= sentences:
                return content
                
            # Simple heuristic: take the first few sentences
            return " ".join(sentences_list[:sentences])
    except Exception as e:
        logger.error(f"Error summarizing webpage {url}: {str(e)}")
        return f"Error: {str(e)}"

@register_tool(
    name="extract_structured_data",
    description="Extract structured data from a webpage using CSS selectors."
)
def extract_structured_data(url: str, selectors: Dict[str, str]) -> Dict[str, List[str]]:
    """Extract structured data from a webpage using CSS selectors.
    
    This function fetches a webpage and extracts structured data based on provided selectors.
    
    Args:
        url: The URL of the webpage to scrape
        selectors: A dictionary mapping data keys to CSS selectors
        
    Returns:
        A dictionary with the extracted data for each key
    """
    try:
        logger.info(f"Extracting structured data from {url}")
        
        # Fetch the HTML content
        html_content = fetch_url(url, as_text=False)
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract data using the provided selectors
        result = {}
        for key, selector in selectors.items():
            elements = soup.select(selector)
            result[key] = [element.get_text(strip=True) for element in elements]
        
        return result
    except Exception as e:
        logger.error(f"Error extracting structured data from {url}: {str(e)}")
        return {"error": str(e)}

@register_tool(
    name="extract_links",
    description="Extract all links from a webpage with optional filtering."
)
def extract_links(url: str, filter_pattern: str = None) -> List[Dict[str, str]]:
    """Extract all links from a webpage with optional filtering.
    
    Args:
        url: The URL of the webpage to extract links from
        filter_pattern: Optional regex pattern to filter links (default: None)
        
    Returns:
        A list of dictionaries containing link information with keys:
        - text: The link text
        - href: The link URL
    """
    try:
        logger.info(f"Extracting links from {url}")
        
        # Fetch the HTML content
        html_content = fetch_url(url, as_text=False)
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract all links
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Convert relative URLs to absolute URLs
            if href.startswith('/'):
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                href = base_url + href
            
            # Apply filter if provided
            if filter_pattern and not re.search(filter_pattern, href):
                continue
                
            links.append({
                "text": a_tag.get_text(strip=True),
                "href": href
            })
        
        return links
    except Exception as e:
        logger.error(f"Error extracting links from {url}: {str(e)}")
        return [{"text": f"Error: {str(e)}", "href": ""}]