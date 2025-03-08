"""Command-line interface for Agnassan.

This module provides a simple CLI for interacting with Agnassan.
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Optional, Dict, Any

from .config import AgnassanConfig, create_default_config
from .routing import Router


class AgnassanCLI:
    """Command-line interface for Agnassan."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.setup_logging()
        self.logger = logging.getLogger("agnassan.cli")
        
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            self.config = AgnassanConfig.from_yaml(config_path)
            self.logger.info(f"Loaded configuration from {config_path}")
        else:
            self.config = create_default_config()
            self.logger.info("Created default configuration")
            
            # Save the default configuration if a path was provided
            if config_path:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                self.config.save_to_yaml(config_path)
                self.logger.info(f"Saved default configuration to {config_path}")
        
        # Create the router
        self.router = Router(self.config)
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.config.log_dir, exist_ok=True)
    
    def setup_logging(self, level: int = logging.INFO):
        """Set up logging configuration."""
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("agnassan.log")
            ]
        )
    
    async def process_query(self, query: str, strategy: str = "auto", image_path: Optional[str] = None) -> str:
        """Process a user query and return the response.
        
        Args:
            query: The text query to process
            strategy: The routing strategy to use
            image_path: Optional path to an image for multimodal queries
        """
        self.logger.info(f"Processing query: {query[:50]}...")
        
        try:
            # Check if image path exists
            if image_path and not os.path.exists(image_path):
                return f"Error: Image file not found: {image_path}"
            
            response = await self.router.route_query(query, strategy, image_path)
            
            # Log the response details
            self.logger.info(f"Response generated using {response.model_name}")
            self.logger.info(f"Tokens used: {response.tokens_used}")
            
            return response.text
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}"
    
    async def interactive_session(self):
        """Start an interactive chat session."""
        print("\n===== Agnassan Chat Interface =====")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'strategy:default', 'strategy:auto', or 'strategy:parallel' to change the routing strategy.")
        print("Type 'image:/path/to/image.jpg' to analyze an image (can be combined with your query).")
        print("=================================\n")
        
        strategy = "auto"
        image_path = None
        
        while True:
            try:
                user_input = input("\nYou: ")
                
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                # Check for strategy change commands
                if user_input.startswith("strategy:"):
                    new_strategy = user_input.split(":")[1].strip()
                    if new_strategy in ["default", "auto", "parallel"]:
                        strategy = new_strategy
                        print(f"Routing strategy changed to: {strategy}")
                    else:
                        print(f"Unknown strategy: {new_strategy}. Available options: default, auto, parallel")
                    continue
                
                # Check for image path commands
                if "image:" in user_input:
                    parts = user_input.split("image:")
                    query = parts[0].strip()
                    image_path = parts[1].strip().split()[0]  # Get the path until the next space
                    
                    # If there's more text after the image path, add it to the query
                    if len(parts[1].strip().split()) > 1:
                        query += " " + " ".join(parts[1].strip().split()[1:])
                    
                    print(f"Using image: {image_path}")
                else:
                    query = user_input
                    image_path = None
                
                # Process the query
                response = await self.process_query(query, strategy, image_path)
                print(f"\nAgnassan: {response}")
                
            except KeyboardInterrupt:
                print("\nSession terminated by user.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def close(self):
        """Close the CLI and free resources."""
        await self.router.close()


async def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Agnassan - An intelligent orchestrator for language models")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process (non-interactive mode)"
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["default", "auto", "parallel"],
        default="auto",
        help="Routing strategy to use"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to an image for multimodal queries"
    )
    
    args = parser.parse_args()
    
    # Create the CLI
    cli = AgnassanCLI(args.config)
    
    # Set logging level based on verbose flag
    if args.verbose:
        cli.setup_logging(logging.DEBUG)
    
    try:
        if args.query:
            # Process a single query in non-interactive mode
            response = await cli.process_query(args.query, args.strategy, args.image)
            print(response)
        else:
            # Start an interactive session
            await cli.interactive_session()
    finally:
        await cli.close()


if __name__ == "__main__":
    asyncio.run(main())