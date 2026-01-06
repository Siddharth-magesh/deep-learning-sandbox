# Binary Positional Encoding

## Overview

Binary Positional Encoding represents positions using their binary representation. Each position in the sequence is encoded as a binary number, where each bit becomes a separate dimension in the encoding. This creates a discrete, structured encoding that explicitly represents position as a number.

## Mathematical Foundation

For a sequence position $pos$ and maximum sequence length requiring $n$ bits:

$$n = \lceil \log_2(\text{seq\_len}) \rceil$$

The encoding for position $pos$ is:

$$PE_{(pos, i)} = \text{bit}_i(\text{binary}(pos))$$

where $\text{bit}_i$ extracts the $i$-th bit from the binary representation of $pos$.

For example:
- Position 0: [0, 0, 0, 0]
- Position 1: [0, 0, 0, 1]
- Position 2: [0, 0, 1, 0]
- Position 3: [0, 0, 1, 1]
- Position 5: [0, 1, 0, 1]

## Key Properties

### Discrete Representation
Unlike sinusoidal or rotary encodings, binary encoding is completely discrete, with only values 0 and 1.

### Logarithmic Dimensionality
The number of dimensions grows logarithmically with sequence length:
- 100 positions: 7 bits
- 1,000 positions: 10 bits
- 10,000 positions: 14 bits

This makes it extremely compact for long sequences.

### Hamming Distance
The difference between two positions can be measured by Hamming distance (number of differing bits), which relates non-linearly to actual distance.

### Hierarchical Structure
Higher-order bits change less frequently, creating a natural hierarchy:
- Least significant bit: alternates every position
- Most significant bit: divides sequence in half

## Implementation Details

The implementation:

1. Calculates required number of bits: $\lceil \log_2(\text{seq\_len}) \rceil$
2. Creates a zero matrix of shape (seq_len, num_bits)
3. For each position, converts to binary and fills corresponding row
4. Pads binary representation with leading zeros to match bit count

## Advantages

- **Extreme compactness**: Logarithmic dimensionality reduces embedding size
- **Explicit encoding**: Directly represents position as a number
- **No computation**: Simple lookup or bit manipulation
- **Hierarchical information**: Different bits capture different scales
- **Deterministic**: Completely predictable pattern

## Limitations

- **Discrete jumps**: No smooth transitions between positions
- **Non-uniform distances**: Hamming distance doesn't match sequence distance
- **Limited expressiveness**: Only 0/1 values, no gradients
- **Poor interpolation**: Adjacent positions can differ significantly in Hamming space
- **Rare usage**: Not commonly used in modern transformers
- **No frequency diversity**: Lacks the rich frequency spectrum of sinusoidal encodings

## Distance Properties

Hamming distance between positions exhibits peculiar properties:
- Positions 0 and 1: differ by 1 bit
- Positions 1 and 2: differ by 2 bits
- Positions 7 and 8: differ by 4 bits

This non-linear relationship to sequence distance can confuse models.

## Use Cases

Potentially useful for:
- Extremely long sequences where dimensionality is critical
- Tasks where exact position matching is important
- Experimental architectures exploring discrete representations
- Combining with other encoding methods (ensemble approaches)
- Educational purposes to understand position encoding concepts

## Theoretical Interest

Binary encoding serves as an interesting baseline for research:
- Demonstrates minimum information-theoretic requirement
- Shows importance of smooth transitions in practice
- Illustrates trade-offs between compactness and expressiveness

## Example Output Characteristics

Visualizations show:
- Block patterns reflecting bit changes
- Least significant bit (rightmost): alternating pattern
- Most significant bit (leftmost): large blocks
- Clear hierarchical structure
- Discrete transitions (sharp boundaries)
- No smooth gradients
