"""Central config for FM training: hyperparameters plus the two swappable
seams (loss function, feature function), both selected by name so a new
implementation can be dropped into losses.py / data.py and picked up here
without touching baseline.py, ablation_features.py, or submit.py."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    k: int = 16
    lr: float = 0.001
    epochs: int = 40
    bs: int = 8192
    patience: int = 4
    loss: str = 'logloss'         # name in losses.LOSSES
    feature_fn: str = 'base'      # name in data.FEATURE_FNS
    sampler: str = 'row'          # 'row' (iid row shuffle) or 'user' (batches
                                   # never split a user's rows) — see baseline.train()

DEFAULT = Config()
