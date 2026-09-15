from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import torch

from rl_lib.envs.env import EnvDetails

@dataclass
class Step:
    action: any
    log_prob: any = None

@runtime_checkable
class Algorithm(Protocol):
    def setup(self, env: EnvDetails, device: torch.device) -> None: ...
    def act(self, state) -> Step: ...
    def timestep_complete(self, timestep: int, state, step: Step, reward, next_state, done) -> None: ...
    def get_stats(self) -> dict: ...
