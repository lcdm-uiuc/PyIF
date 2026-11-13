
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class KSGConfig:
    k_neighbors: int = 1
    backend: str = "cpu"
    safety_check: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.k_neighbors, Integral)
            or isinstance(self.k_neighbors, bool)
            or self.k_neighbors < 1
        ):
            raise ValueError("k_neighbors must be a positive integer")

        object.__setattr__(
            self, "k_neighbors", int(self.k_neighbors)
        )

        if self.backend not in ("cpu", "gpu"):
            raise ValueError("backend must be 'cpu' or 'gpu'")

        if type(self.safety_check) is not bool:
            raise TypeError("safety_check must be a boolean")
