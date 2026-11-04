
from dataclasses import dataclass


@dataclass(frozen=True)
class KSGConfig:
    k_neighbors: int = 1
    backend: str = "cpu"
    safety_check: bool = False

    def __post_init__(self):
        if type(self.k_neighbors) is not int or self.k_neighbors < 1:
            raise ValueError("k_neighbors must be a positive integer")

        if self.backend not in ("cpu", "gpu"):
            raise ValueError("backend must be 'cpu' or 'gpu'")

        if type(self.safety_check) is not bool:
            raise TypeError("safety_check must be a boolean")
