# Positional Encoding Methods - Comparison Guide

## Quick Overview

| Method | Type | Parameters | Dimensions | Best For | Extrapolation |
|--------|------|------------|------------|----------|---------------|
| Sinusoidal | Additive | 0 | d_model | General use | Good |
| Rotary | Multiplicative | 0 | d_model | Long sequences | Excellent |
| ALiBi | Attention bias | 0 | N/A | Length extrapolation | Excellent |
| Binary | Additive | 0 | log₂(seq_len) | Compact encoding | Poor |
| Index | Additive | 0 | 1 | With learned embeddings | Poor |

## Detailed Comparison

### Computational Complexity

**Sinusoidal**: O(seq_len × d_model)
- Can be precomputed and cached
- One-time computation per sequence length

**Rotary**: O(seq_len × d_model) per forward pass
- Applied to queries and keys in attention
- Cannot be fully precomputed (depends on input)

**ALiBi**: O(seq_len²) bias matrix
- Very lightweight (just slopes + positions)
- Added to attention scores

**Binary**: O(seq_len × log₂(seq_len))
- Extremely efficient
- Minimal memory footprint

**Index**: O(seq_len)
- Trivial computation
- Often used as embedding input

### Memory Requirements

**Lowest**: Index (seq_len × 1)
**Low**: Binary (seq_len × log₂(seq_len))
**Medium**: Sinusoidal, Rotary (seq_len × d_model)
**Attention-dependent**: ALiBi (num_heads × seq_len × seq_len or just slopes)

### Length Extrapolation Performance

**Best**: ALiBi, Rotary
- Designed specifically for extrapolation
- Proven effective on sequences 4-10× longer than training

**Good**: Sinusoidal
- Can handle some extrapolation
- May degrade on very long sequences

**Poor**: Binary, Index
- Struggle with sequences longer than training
- Normalized versions fail when range exceeds training bounds

### Relative vs. Absolute Position

**Strong Relative**: RoPE, ALiBi
- Explicitly encode relative distances
- Natural for learning distance-based patterns

**Learnable Relative**: Sinusoidal
- Contains relative position information
- Model must learn to extract it

**Primarily Absolute**: Binary, Index
- Focus on absolute position
- Relative relationships not inherent

### Use in Modern Models

**Sinusoidal**: 
- Original Transformer, T5, many classical transformers
- Still widely used baseline

**Rotary**: 
- GPT-Neo, GPT-J, LLaMA, PaLM
- Increasingly popular for large language models

**ALiBi**: 
- BLOOM, MPT
- Chosen specifically for extrapolation needs

**Binary/Index**: 
- Rare in production
- Mostly research and baseline comparisons

## Selection Guide

### Choose Sinusoidal when:
- Building standard transformer architecture
- Need proven, reliable approach
- Interpretability is important
- Working with moderate sequence lengths
- Want parameter-free solution

### Choose Rotary when:
- Building large language models
- Long-range dependencies are critical
- Need excellent extrapolation
- Willing to modify attention mechanism
- Following modern LLM practices

### Choose ALiBi when:
- Extrapolation is top priority
- Variable sequence lengths in deployment
- Want simplest possible implementation
- Training efficiency matters
- No positional embeddings desired

### Choose Binary when:
- Sequence length is extremely long (millions)
- Memory is severely constrained
- Experimenting with discrete representations
- Combining with learned embeddings

### Choose Index when:
- Using learned positional embeddings
- Building baseline for comparison
- Need simple debugging tool
- Implementing curriculum learning

## Performance Characteristics

### Training Speed
**Fastest**: ALiBi (no embeddings), Index
**Fast**: Binary, Sinusoidal (precomputed)
**Moderate**: Rotary (computed per batch)

### Inference Speed
**Fastest**: ALiBi, Index
**Fast**: Binary, Sinusoidal (cached)
**Moderate**: Rotary (recomputed)

### Model Quality (sequence modeling)
**Best**: Rotary, ALiBi (task-dependent)
**Good**: Sinusoidal
**Limited**: Binary, Index (without learned embeddings)

## Hybrid Approaches

Many modern systems combine methods:

**Rotary + ALiBi**: Some models use both for complementary benefits

**Sinusoidal + Learned**: Add learnable weights to sinusoidal encodings

**Index + Embedding**: Use index as input to learned embedding layer

## Research Trends

**Current Focus**: 
- Extending context length (Rotary variations, ALiBi)
- Reducing computational cost
- Improving extrapolation

**Emerging Directions**: 
- Adaptive position encodings
- Task-specific position learning
- Position encodings for non-sequential data

## Practical Recommendations

**Default Choice**: Sinusoidal (proven, reliable)

**Modern LLMs**: Rotary (state-of-the-art for language)

**Long Context**: ALiBi or Rotary with extensions

**Research**: Try multiple, compare empirically

**Production**: Match your training setup (don't switch methods post-training)
