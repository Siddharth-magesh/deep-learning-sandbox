# Inference Guide

## Text Generation with GPT-2

### Quick Start

#### Interactive Mode

Start an interactive session for continuous text generation:

```bash
cd d:\ai_research_learning
python -m generative-pretrained-transformer-2.src.inference \
    --model_path generative-pretrained-transformer-2/checkpoints/best_model.pth \
    --interactive
```

**Usage:**
1. Enter your prompt when asked
2. Watch the model generate text in real-time (streaming)
3. Type 'quit' or 'exit' to stop
4. Press Ctrl+C to interrupt

#### Single Generation

Generate text for a single prompt:

```bash
python -m generative-pretrained-transformer-2.src.inference \
    --model_path generative-pretrained-transformer-2/checkpoints/best_model.pth \
    --prompt "Once upon a time in a distant galaxy"
```

## Generation Parameters

### Temperature

Controls randomness of predictions:

```bash
--temperature 0.7  # More focused, conservative
--temperature 1.0  # Balanced (default)
--temperature 1.3  # More creative, random
```

**Effect:**
- Low (0.1-0.7): Deterministic, repetitive
- Medium (0.7-1.0): Balanced
- High (1.0-2.0): Creative, chaotic

**Recommendation:** 0.8 for most tasks

### Top-k Sampling

Only sample from the k most likely next tokens:

```bash
--top_k 50   # Consider top 50 tokens (default)
--top_k 10   # More conservative
--top_k 100  # More diverse
```

**Effect:**
- Lower k: More focused, less diverse
- Higher k: More diverse, potentially incoherent

**Recommendation:** 40-50 for balanced output

### Top-p (Nucleus) Sampling

Sample from smallest set of tokens with cumulative probability > p:

```bash
--top_p 0.95  # Default
--top_p 0.9   # More focused
--top_p 0.99  # More diverse
```

**Effect:**
- Lower p: More conservative
- Higher p: More diverse

**Recommendation:** 0.92-0.95 for quality output

### Repetition Penalty

Penalize repeated tokens:

```bash
--repetition_penalty 1.0  # No penalty
--repetition_penalty 1.2  # Moderate (default)
--repetition_penalty 1.5  # Strong penalty
```

**Effect:**
- 1.0: No penalty (may repeat)
- 1.2-1.3: Balanced
- >1.5: Strongly avoid repetition

**Recommendation:** 1.2 for natural text

### Max Tokens

Maximum number of new tokens to generate:

```bash
--max_new_tokens 50   # Short response
--max_new_tokens 100  # Default
--max_new_tokens 500  # Long response
```

**Note:** Limited by context window (1024 tokens total)

### Sampling vs Greedy

**Sampling (default):**
```bash
# Uses temperature, top_k, top_p
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "Hello world"
```

**Greedy Decoding:**
```bash
# Always picks most likely token
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "Hello world" \
    --no_sample
```

**When to use:**
- Sampling: Creative writing, variety needed
- Greedy: Deterministic output, factual tasks

### Streaming

**Enabled (default):**
```bash
# Tokens appear as they're generated
--stream
```

**Disabled:**
```bash
# Full text appears at once
--no_stream
```

## Usage Examples

### Creative Writing

```bash
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "In a world where magic exists" \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.95 \
    --max_new_tokens 200
```

### Technical Writing

```bash
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "The algorithm works by" \
    --temperature 0.7 \
    --top_k 40 \
    --repetition_penalty 1.3 \
    --max_new_tokens 150
```

### Dialogue Generation

```bash
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "Person A: How are you?\nPerson B:" \
    --temperature 0.8 \
    --top_p 0.92 \
    --max_new_tokens 100
```

### Code Generation

```bash
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "def fibonacci(n):" \
    --temperature 0.5 \
    --top_k 30 \
    --repetition_penalty 1.1 \
    --max_new_tokens 150
```

### Story Continuation

```bash
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "The old man walked into the forest and found" \
    --temperature 1.0 \
    --top_k 60 \
    --top_p 0.95 \
    --max_new_tokens 300
```

## Device Selection

### Use GPU (default)

```bash
--device cuda
```

### Force CPU

```bash
--device cpu
```

**Note:** GPU is 10-50x faster for inference

## Advanced Techniques

### Multi-turn Conversation

Concatenate previous context:

```bash
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "User: What is AI?\nAssistant: AI stands for Artificial Intelligence.\nUser: How does it work?\nAssistant:" \
    --max_new_tokens 100
```

### Controlled Generation

Use specific formats:

```bash
python -m generative-pretrained-transformer-2.src.inference \
    --model_path checkpoints/best_model.pth \
    --prompt "Title: The Future of AI\nSummary:" \
    --temperature 0.7 \
    --max_new_tokens 150
```

### Batched Prompts

For multiple prompts, use a script:

```python
from generative-pretrained-transformer-2.src.inference import TextGenerator
from generative-pretrained-transformer-2.src.config import InferenceConfig

generator = TextGenerator('checkpoints/best_model.pth')
config = InferenceConfig(max_new_tokens=100, temperature=0.8)

prompts = [
    "Once upon a time",
    "In the future",
    "The scientist discovered"
]

for prompt in prompts:
    generator.generate_text(prompt, config)
```

## Parameter Tuning Guide

### For High Quality

```
temperature: 0.7-0.8
top_k: 40-50
top_p: 0.92-0.95
repetition_penalty: 1.2-1.3
```

### For Creativity

```
temperature: 0.9-1.2
top_k: 50-80
top_p: 0.95-0.98
repetition_penalty: 1.1-1.2
```

### For Consistency

```
temperature: 0.5-0.7
top_k: 30-40
top_p: 0.9-0.92
repetition_penalty: 1.3-1.5
```

### For Speed

```
max_new_tokens: 50-100
no_sample: True (greedy)
stream: False
```

## Common Issues

### Repetitive Output

**Solutions:**
- Increase repetition_penalty to 1.3-1.5
- Increase top_k to 60-80
- Adjust temperature to 0.8-1.0

### Incoherent Output

**Solutions:**
- Decrease temperature to 0.6-0.8
- Decrease top_k to 30-40
- Decrease top_p to 0.9

### Slow Generation

**Solutions:**
- Use GPU (--device cuda)
- Reduce max_new_tokens
- Use --no_stream
- Ensure model is in eval mode

### Cut-off Text

**Solutions:**
- Increase max_new_tokens
- Model may have reached natural end
- Check if EOS token was generated

## Programmatic Usage

### Python API

```python
import torch
from generative-pretrained-transformer-2.src.inference import TextGenerator
from generative-pretrained-transformer-2.src.config import InferenceConfig

generator = TextGenerator('checkpoints/best_model.pth', device='cuda')

config = InferenceConfig(
    max_new_tokens=150,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    stream=True
)

generator.generate_text("Your prompt here", config)
```

### Interactive in Python

```python
generator = TextGenerator('checkpoints/best_model.pth')
config = InferenceConfig()
generator.interactive_mode(config)
```

### Custom Generation Loop

```python
from generative-pretrained-transformer-2.src.model import GPT2Model
from transformers import GPT2Tokenizer
import torch

model = GPT2Model.from_checkpoint('checkpoints/best_model.pth')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "Your prompt"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

with torch.no_grad():
    for token in model.generate(input_ids, max_new_tokens=100):
        print(tokenizer.decode(token.tolist()), end='', flush=True)
```

## Performance Benchmarks

**Generation Speed (NVIDIA GeForce RTX):**
- 50 tokens: ~2-3 seconds
- 100 tokens: ~4-6 seconds
- 500 tokens: ~20-30 seconds

**Memory Usage:**
- Model: ~500MB
- Generation: +100-200MB per batch
- Total: ~700MB minimum

## Best Practices

1. **Start with defaults** - Adjust only if needed
2. **Use streaming** - Better user experience
3. **Monitor quality** - Adjust parameters iteratively
4. **Use GPU** - Significantly faster
5. **Experiment** - Different tasks need different settings
6. **Save configs** - Document successful parameter combinations
7. **Test prompts** - Clear, specific prompts work best
8. **Context matters** - Include relevant context in prompt
9. **Check output** - Always validate generated text
10. **Iterate** - Refine prompts and parameters together
