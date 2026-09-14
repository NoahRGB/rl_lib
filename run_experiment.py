from dataclasses import dataclass, field
from typing import Any
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING
import torch

from rl_lib.experiment.runner import run_experiment

@dataclass
class EnvConf:
    _target_: str = MISSING
    env_id: str = MISSING
    num_envs: int = MISSING

@dataclass
class LoggerConf:
    _target_: str = MISSING
    files_log_dir: str = MISSING
    tensorboard_log_dir: str = MISSING
    use_tensorboard: bool = MISSING
    use_files: bool = MISSING
    print_progress: bool = MISSING

@dataclass
class ExperimentConf:
    env: EnvConf = field(default_factory=EnvConf)
    logger: LoggerConf = field(default_factory=LoggerConf)
    alg: Any = MISSING
    title: str = MISSING
    timesteps: int = MISSING
    device: str = MISSING
    seed: int = MISSING

cs = ConfigStore.instance()
cs.store(name="experiment_conf", node=ExperimentConf)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)

    agent = hydra.utils.instantiate(cfg.alg)

    env = hydra.utils.instantiate(cfg.env, _partial_=True)
    env = env(seed=cfg.seed)

    logger = hydra.utils.instantiate(cfg.logger, _partial_=True)
    logger = logger(title=cfg.title)

    run_experiment(agent, env, logger, cfg.timesteps, cfg.seed, device)

if __name__ == "__main__":
    main()
