import torch
import numpy as np

from rl_lib.algs.alg import Step
from rl_lib.envs.env import EnvDetails
from rl_lib.utils.networks import ActorCriticNetwork
from rl_lib.utils.buffers import OnPolicyBuffer
from rl_lib.utils.heads import CategoricalHead

class A2C:

    def __init__(self, lr: float, tmax: int, gamma: float, lam: float, 
                 cgn: float, entropy_weight: float, value_weight: float, 
                 net_architecture: dict, load_path: str = None):
        self.lr = lr
        self.tmax = tmax
        self.gamma = gamma
        self.lam = lam
        self.cgn = cgn
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.net_architecture = net_architecture
        self.load_path = load_path

    def setup(self, env: EnvDetails, device: torch.device) -> None:
        self.env = env
        self.device = device
        self.stats = {}

        self.network = ActorCriticNetwork(env, self.net_architecture).to(device)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.buffer = OnPolicyBuffer(self.tmax, env.num_envs, env.state_space.shape, env.action_space.shape)
        self.is_continuous = self.network.get_head_type() is not CategoricalHead

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, map_location=device)
            self.network.load_state_dict(checkpoint["net"])
            self.optim.load_state_dict(checkpoint["optim"])

    def learn(self):
        full_batch = self.buffer.get(self.device)
        s_flat = full_batch.states.view(-1, *self.env.state_space.shape)

        if self.is_continuous:
            (mu, sigma), state_values = self.network(s_flat)
            mu = mu.view(self.tmax, self.env.num_envs, -1)
            sigma = sigma.view(self.tmax, self.env.num_envs, -1)
            distribution = torch.distributions.Normal(mu, sigma)
            chosen_log_probs = distribution.log_prob(full_batch.actions).sum(dim=-1)
        else:
            logits, state_values = self.network(s_flat)
            distribution = torch.distributions.Categorical(logits=logits.view(self.tmax, self.env.num_envs, -1))
            chosen_log_probs = distribution.log_prob(full_batch.actions)

        _, final_state_values = self.network(full_batch.next_states[-1].view(-1, *self.env.state_space.shape))

        state_values = state_values.view(self.tmax, self.env.num_envs)
        final_state_values = final_state_values.view(self.env.num_envs) # (num_envs,)

        with torch.no_grad():
            advantages, returns = full_batch.gae(state_values, final_state_values, self.gamma, self.lam, self.device)

        entropy_bonus = distribution.entropy().mean()
        policy_loss = -(chosen_log_probs * advantages).mean() - (self.entropy_weight * entropy_bonus)
        state_value_loss = torch.nn.functional.mse_loss(state_values, returns)
        combined_loss = policy_loss + self.value_weight * state_value_loss

        self.optim.zero_grad()
        combined_loss.backward()
        if self.cgn is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cgn)
        self.optim.step()

    def act(self, state) -> Step:
        with torch.no_grad():
            actor_out, _ = self.network(torch.from_numpy(state).float().to(self.device))

            if self.is_continuous:
                mu, sigma = actor_out
                distribution = torch.distributions.Normal(mu, sigma)
                action = distribution.sample()
                log_prob = distribution.log_prob(action).sum(dim=-1)
            else:
                logits = actor_out
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)

            return Step(action=action.cpu().numpy(), log_prob=log_prob.cpu().numpy())

    def timestep_complete(self, timestep: int, state, step: Step, reward, next_state, done) -> None:
        self.buffer.add(state, step.action, reward, next_state, done, step.log_prob)
        if self.buffer.is_full():
            self.learn()
            self.buffer.clear()

    def get_stats(self) -> dict:
        return self.stats
