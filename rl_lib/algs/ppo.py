import torch
import numpy as np

from rl_lib.algs.alg import Step
from rl_lib.envs.env import EnvDetails
from rl_lib.utils.networks import ActorCriticNetwork
from rl_lib.utils.heads import CategoricalHead

class PPO:

    def __init__(self, lr: float, net_architecture: dict):
        self.lr = lr
        self.net_architecture = net_architecture

    def setup(self, env: EnvDetails, device: torch.device) -> None:
        self.env = env
        self.device = device

        self.network = ActorCriticNetwork(env, 0.001, self.net_architecture)
        self.is_continuous = self.network.get_head_type() is not CategoricalHead

    def act(self, state) -> Step:
        with torch.no_grad():
            action, _ = self.network(torch.from_numpy(state).float().to(self.device))

            if self.is_continuous:
                ...
            else:
                distribution = torch.distributions.Categorical(logits=action)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
                return Step(action=action.cpu().numpy(), log_prob=log_prob.cpu().numpy())

    def timestep_complete(self, state, step: Step, reward, next_state, done) -> None:
        ...

    def get_stats(self) -> dict:
        ...
