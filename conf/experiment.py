from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EnvConf:
    _target_: str = "envs.gym_wrapper.GymEnv"
    env_id: str = "CartPole-v1"
    num_envs: int = 8
    max_episode_steps: Optional[int] = None

@dataclass
class ExperimentConf:
    env: EnvConf = field(default_factory=EnvConf)
    agent: AgentConf = field(default_factory=AgentConf)
    seed: int = 0
    total_steps: int = 1000000
    device: str = "cuda"
