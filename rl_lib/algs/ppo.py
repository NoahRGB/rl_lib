import torch
import numpy as np

from rl_lib.algs.alg import Step
from rl_lib.envs.env import EnvDetails
from rl_lib.utils.networks import ActorCriticNetwork
from rl_lib.utils.buffers import OnPolicyBuffer
from rl_lib.utils.heads import CategoricalHead

class PPO:

    def __init__(self, lr: float, tmax: int, gamma: float, lam: float, 
                 epochs: int, minibatch_size: int, epsilon: float, cgn: float,
                 entropy_weight: float, value_weight: float, net_architecture: dict,
                 load_path: str = None):
        self.lr = lr
        self.tmax = tmax
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.cgn = cgn
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size
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

        with torch.no_grad():
            _, state_values = self.network(full_batch.states.view(-1, *self.env.state_space.shape)) # (tmax, num_envs, 1)
            _, final_state_values = self.network(full_batch.next_states[-1].view(-1, *self.env.state_space.shape)) # (tmax, num_envs, 1)

            state_values = state_values.view(self.tmax, self.env.num_envs)
            final_state_values = final_state_values.view(self.env.num_envs) # (num_envs,)

            advantages, returns = full_batch.gae(state_values, final_state_values, self.gamma, self.lam, self.device)
            flat_batch = full_batch.flatten()
            flat_advantages = advantages.view(self.tmax*self.env.num_envs)
            flat_returns = returns.view(self.tmax*self.env.num_envs)

        for epoch in range(self.epochs):

            total_batch_size = self.tmax * self.env.num_envs
            all_indices = np.arange(total_batch_size)
            np.random.shuffle(all_indices)
            for minibatch_start in range(0, total_batch_size, self.minibatch_size):
                minibatch_indices = all_indices[minibatch_start:minibatch_start + self.minibatch_size]
                minibatch = flat_batch.get_minibatch(minibatch_indices)
                minibatch_advantages = flat_advantages[minibatch_indices].detach()
                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                if self.is_continuous:
                    (minibatch_mu, minibatch_sigma), minibatch_state_values = self.network(minibatch.states)
                    distribution = torch.distributions.Normal(minibatch_mu, minibatch_sigma)
                    chosen_log_probs = distribution.log_prob(minibatch.actions).sum(dim=-1)
                else:
                    minibatch_logits, minibatch_state_values = self.network(minibatch.states)
                    distribution = torch.distributions.Categorical(logits=minibatch_logits)
                    chosen_log_probs = distribution.log_prob(minibatch.actions)

                ratios = torch.exp(chosen_log_probs - minibatch.log_probs)
                clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                surrogate_obj = ratios * minibatch_advantages
                clipped_surrogate_obj = clipped_ratios * minibatch_advantages

                entropy_bonus = distribution.entropy().mean()
                policy_loss = -torch.min(surrogate_obj, clipped_surrogate_obj).mean() - (self.entropy_weight * entropy_bonus)
                state_value_loss = torch.nn.functional.mse_loss(minibatch_state_values.squeeze(-1), flat_returns[minibatch_indices].detach())
                combined_loss = policy_loss + self.value_weight * state_value_loss

                self.optim.zero_grad()
                combined_loss.backward()
                if self.cgn is not None:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cgn)
                self.optim.step()

                self.stats = {
                    "metrics": {
                        "policy_loss": policy_loss.item(),
                        "value_loss": state_value_loss.item(),
                        "entropy_bonus": entropy_bonus.item(),
                        "mean_advantage": minibatch_advantages.mean().item(),
                        "mean_return": flat_returns[minibatch_indices].mean().item()
                    }
                }
                
        self.stats["network"] = {"net": self.network.state_dict(), "optim": self.optim.state_dict()}

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
