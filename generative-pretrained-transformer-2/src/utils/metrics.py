import math


def calculate_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))
