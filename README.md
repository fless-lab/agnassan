# Agnassan: Advanced AI Orchestration System

Agnassan is a powerful AI orchestration system that leverages multiple open-source models with sophisticated reasoning techniques to provide enhanced problem-solving capabilities.

## Key Features

### Advanced Reasoning Techniques

Agnassan implements several advanced reasoning techniques:

1. **Chain of Thought (CoT)** - Breaks down complex problems into sequential steps, making the reasoning process explicit and easier to follow.

2. **Tree of Thought (ToT)** - Explores multiple reasoning paths simultaneously and selects the most promising one, allowing for more comprehensive problem-solving.

3. **Parallel Thought Chains** - Applies multiple reasoning techniques in parallel and synthesizes their results, leveraging different cognitive approaches for more robust solutions.

4. **Iterative Loops** - Applies a sequence of different reasoning techniques iteratively, with each technique building upon the results of the previous one, creating increasingly refined answers.

5. **ReAct (Reasoning + Acting)** - Combines reasoning with the ability to take actions, enabling more interactive and dynamic problem-solving capabilities.

6. **Meta-Critique** - Critically evaluates and improves the initial response, leading to more accurate and well-considered answers.

### Model Integration

Agnassan supports a wide range of models:

- **Open-source models**: Mistral-7B, Phi-2, Llama-3-8B, Gemma-7B, MPT-7B
- **Commercial models**: GPT-4o, Claude-3-Opus (optional)

### Intelligent Routing

The system includes an intelligent routing mechanism that:

1. Analyzes queries to determine the most appropriate model(s) and reasoning technique
2. Supports parallel execution across multiple models
3. Dynamically selects the best reasoning approach based on the query type

## Installation

```bash
pip install -r requirements.txt
```

## Using Models from Hugging Face API

Agnassan supports using models directly from the Hugging Face API. Here's how to configure and use them:

```python
from agnassan.config import ModelConfig
from agnassan.models import create_model_interface
import asyncio

# Configure a Hugging Face model
config = ModelConfig(
    name="gpt2",  # You can use any model available on Hugging Face
    provider="huggingface",
    api_key_env="HUGGINGFACE_API_KEY",  # Set this environment variable
    parameters={
        "model": "gpt2",
        "temperature": 0.7,
        "max_tokens": 100
    },
    strengths=["general_knowledge", "creative"]
)

# Create the model interface
model = create_model_interface(config)

# Generate text
async def generate_text():
    prompt = "Once upon a time"
    response = await model.generate(prompt)
    print(f"Generated text: {response.text}")
    print(f"Tokens used: {response.tokens_used}")

# Run the async function
asyncio.run(generate_text())
```

## Downloading and Using Local Models

For better performance and to avoid API costs, you can download models and use them locally:

```python
from agnassan.config import ModelConfig
from agnassan.models import create_model_interface
import asyncio
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to download a model
def download_model(model_name, save_path):
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Download model and tokenizer
    print(f"Downloading {model_name} to {save_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save to local directory
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Model {model_name} downloaded successfully!")

# Download a model (only needs to be done once)
model_name = "gpt2"  # You can use any model from Hugging Face
save_path = Path("./models/gpt2")

if not save_path.exists():
    download_model(model_name, save_path)

# Configure the local model
config = ModelConfig(
    name="local-gpt2",
    provider="local",
    parameters={
        "model_path": str(save_path),
        "temperature": 0.7,
        "max_tokens": 100
    },
    strengths=["general_knowledge", "creative"]
)

# Create the model interface
model = create_model_interface(config)

# Generate text
async def generate_text():
    prompt = "The future of AI is"
    response = await model.generate(prompt)
    print(f"Generated text: {response.text}")
    print(f"Tokens used: {response.tokens_used}")

# Run the async function
asyncio.run(generate_text())
```

## Supported Model Providers

Agnassan supports the following model providers:

- **OpenAI**: GPT models (requires `openai` package)
- **Anthropic**: Claude models (requires `anthropic` package)
- **Hugging Face**: Any model on Hugging Face Hub (requires `huggingface_hub` package)
- **Cohere**: Cohere models (requires `cohere` package)
- **Grok**: Grok models (requires `aiohttp` package)
- **Replicate**: Models hosted on Replicate (requires `replicate` package)
- **Local**: Downloaded models (requires `transformers` and `torch` packages)

## Configuration

You can configure multiple models in a YAML file:

```yaml
models:
  - name: mistral-7b
    provider: local
    parameters:
      model_path: ./models/mistral-7b
      context_length: 4096
      temperature: 0.7
    strengths:
      - general_knowledge
      - coding
    cost_per_token: 0.0
  
  - name: gpt-4o
    provider: openai
    api_key_env: OPENAI_API_KEY
    parameters:
      model: gpt-4o
      temperature: 0.7
      max_tokens: 4096
    strengths:
      - complex_reasoning
      - creative
      - coding
    cost_per_token: 0.00001

default_model: mistral-7b
log_dir: ./logs
routing_strategy: rule_based
```

Load the configuration with:

```python
from agnassan.config import AgnassanConfig

config = AgnassanConfig.from_yaml("config.yaml")
```

## Running Tests

To run the tests that demonstrate how to use different model interfaces:

```bash
python -m tests.test_models
```

Or use pytest:

```bash
pytest tests/test_models.py
```