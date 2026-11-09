
from dataclasses import dataclass, field


@dataclass(frozen=True)
class InferenceResult:
    status: str = "not_computed"
    method: str | None = None
    n_surrogates: int = 0
    surrogate_mean: float | None = None
    corrected_te: float | None = None
    p_value: float | None = None



@dataclass(frozen=True)
class TEResult:
    raw_te: float
    units: str
    n_input: int
    n_history_rows: int
    resolved_settings: dict
    software_version: str
    data_hashes: dict

    code_revision: str | None = None
    working_tree_dirty: bool | None = None
    random_seed: int | None = None

    inference: InferenceResult = field(
        default_factory=InferenceResult
    )
