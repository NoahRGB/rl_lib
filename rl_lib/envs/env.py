from typing import Protocol
from dataclasses import dataclass

from rl_lib.utils.spaces import Discrete, Continuous

@dataclass
class EnvDetails:
    num_envs: int
    state_space: Discrete|Continuous
    action_space: Discrete|Continuous

class Environment(Protocol):
    details: EnvDetails
    seed: int|None
    
    def reset(self) -> any: ...
    def get_start_states(self) -> any: ...
    def step(self, action) -> tuple: ...

