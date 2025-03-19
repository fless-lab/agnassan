"""Debug script to test loading the Phi-2 model."""

import logging
import sys
from agnassan.config import ModelConfig, create_default_config
from agnassan.models import create_model_interface

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def debug_phi2_model():
    try:
        print("Creating model config for Phi-2...")
        
        # Create a simple model config for Phi-2
        config = ModelConfig(
            name="phi-2",
            provider="local",
            parameters={
                "model_id": "phi-2",
                "model_path": "./models/phi-2",
                "temperature": 0.7,
                "load_in_8bit": True,
                "trust_remote_code": True
            },
            strengths=["reasoning", "math", "coding"]
        )
        
        print("Creating model interface...")
        model = create_model_interface(config)
        print(f"Model interface created: {type(model)}")
        
        print("Testing model generation...")
        import asyncio
        response = asyncio.run(model.generate("Hello, how are you?"))
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_phi2_model()