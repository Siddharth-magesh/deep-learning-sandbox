from dataclasses import dataclass

@dataclass
class RuntimeConfig:
    seed: int = 42
    output_dir: str = "./densely-connected-convolutional-networks/outputs"
    experiment_name: str = "densenet201_baseline"

