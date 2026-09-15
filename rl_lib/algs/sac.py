import torch
import numpy as np

from rl_lib.algs.alg import Step
from rl_lib.envs.env import EnvDetails
from rl_lib.utils.networks import QNetwork, ActorNetwork
from rl_lib.utils.buffers import OffPolicyBuffer
from rl_lib.utils.spaces import Discrete

class SAC:

    def __init__(self, lr: float, replay_size: int, target_update_interval: int,
                 update_freq: int, gradient_steps: int, minibatch_size: int,
                 gamma: float, alpha_start: float, auto_alpha: bool, target_factor: float,
                 noise_factor: float, cgn: float, warmup_steps: int,
                 net_architecture: dict, load_path: str = None):

        self.lr = lr
        self.replay_size = replay_size
        self.target_update_interval = target_update_interval
        self.update_freq = update_freq
        self.gradient_steps = gradient_steps
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.cgn = cgn
        self.net_architecture = net_architecture
        self.warmup_steps = warmup_steps
        self.alpha_start = alpha_start
        self.auto_alpha = auto_alpha
        self.target_factor = target_factor
        self.noise_factor = noise_factor
        self.load_path = load_path

    def setup(self, env: EnvDetails, device: torch.device) -> None:
        self.env = env
        self.device = device
        self.stats = {}
        assert self.env.num_envs == 1

        self.is_continuous = type(self.env.action_space) != Discrete

        self.buffer = OffPolicyBuffer(self.replay_size, env.num_envs, env.state_space.shape, env.action_space.shape)

        self.actor = ActorNetwork(env, self.net_architecture, is_deterministic=False).to(self.device)
        self.qfunc1 = QNetwork(env, self.net_architecture, input_shape=(env.state_space.shape[0]+env.action_space.shape[0],)).to(self.device)
        self.target_qfunc1 = QNetwork(env, self.net_architecture, input_shape=(env.state_space.shape[0]+env.action_space.shape[0],)).to(self.device)
        self.qfunc2 = QNetwork(env, self.net_architecture, input_shape=(env.state_space.shape[0]+env.action_space.shape[0],)).to(self.device)
        self.target_qfunc2 = QNetwork(env, self.net_architecture, input_shape=(env.state_space.shape[0]+env.action_space.shape[0],)).to(self.device)

        self.target_qfunc1.load_state_dict(self.qfunc1.state_dict())
        self.target_qfunc2.load_state_dict(self.qfunc2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.qfunc1_optim = torch.optim.Adam(self.qfunc1.parameters(), lr=self.lr)
        self.qfunc2_optim = torch.optim.Adam(self.qfunc2.parameters(), lr=self.lr)

        self.entropy_target = -env.action_space.shape[0] # paper says this should be -dim(A) (e.g. -6 for HalfCheetah)
        if self.auto_alpha:
            self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(self.alpha_start)).to(self.device)).to(self.device)
            self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = self.alpha_start

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.qfunc1.load_state_dict(checkpoint["qfunc1"])
            self.qfunc2.load_state_dict(checkpoint["qfunc2"])
            self.actor_optim.load_state_dict(checkpoint["actor_optim"])
            self.qfunc1_optim.load_state_dict(checkpoint["qfunc1_optim"])
            self.qfunc2_optim.load_state_dict(checkpoint["qfunc2_optim"])

    def learn(self):
        full_batch, batch_size = self.buffer.sample(self.minibatch_size, self.device)
    
        if batch_size == self.minibatch_size:

            qnet_input = torch.concat([full_batch.states, full_batch.actions], dim=-1)
            qfunc1_vals = self.qfunc1(qnet_input).squeeze(-1)
            qfunc2_vals = self.qfunc2(qnet_input).squeeze(-1)

            with torch.no_grad():
                # prepare some "fresh" CURRENT/ONPOLICY actions based on sprimes
                fresh_mu, fresh_sigma = self.actor(full_batch.next_states)
                fresh_dists = torch.distributions.Normal(fresh_mu, fresh_sigma)
                fresh_raw_actions = fresh_dists.rsample()
                fresh_actions = torch.tanh(fresh_raw_actions)
                fresh_action_network_input = torch.concat([full_batch.next_states, fresh_actions], dim=-1)
                fresh_actions_log_probs = fresh_dists.log_prob(fresh_raw_actions).sum(-1) - torch.log(1 - fresh_actions.pow(2) + 1e-6).sum(-1)

                # calculate Q targets
                min_qvals = torch.min(self.target_qfunc1(fresh_action_network_input), self.target_qfunc2(fresh_action_network_input)).squeeze(-1)
                if self.auto_alpha:
                    qfunc_targets = full_batch.rewards + self.gamma * full_batch.masks * (min_qvals - self.log_alpha.exp().detach() * fresh_actions_log_probs) 
                else:
                    qfunc_targets = full_batch.rewards + self.gamma * full_batch.masks * (min_qvals - self.alpha * fresh_actions_log_probs)

            # backprop + SGD for both qfuncs
            qfunc1_loss = torch.nn.functional.mse_loss(qfunc1_vals, qfunc_targets)
            qfunc2_loss = torch.nn.functional.mse_loss(qfunc2_vals, qfunc_targets)
            self.qfunc1_optim.zero_grad()
            qfunc1_loss.backward()
            self.qfunc1_optim.step()
            self.qfunc2_optim.zero_grad()
            qfunc2_loss.backward()
            self.qfunc2_optim.step()

            # update actor

            fresh_mu, fresh_sigma = self.actor(full_batch.states) 
            fresh_dists = torch.distributions.Normal(fresh_mu, fresh_sigma)
            fresh_raw_actions = fresh_dists.rsample() 
            fresh_actions = torch.tanh(fresh_raw_actions) 
            fresh_action_network_input = torch.concat([full_batch.states, fresh_actions], dim=-1) 
            fresh_actions_log_probs = fresh_dists.log_prob(fresh_raw_actions).sum(-1) - torch.log(1 - fresh_actions.pow(2) + 1e-6).sum(-1) 

            min_qvals = torch.min(self.qfunc1(fresh_action_network_input), self.qfunc2(fresh_action_network_input)).squeeze(-1) 

            if self.auto_alpha:
                policy_loss = -(min_qvals - self.log_alpha.exp().detach() * fresh_actions_log_probs).mean()
            else:
                policy_loss = -(min_qvals - self.alpha * fresh_actions_log_probs).mean()
                
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

    def act(self, state) -> Step:
        with torch.no_grad():
            mu, sigma = self.actor(torch.from_numpy(state).to(self.device))
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.rsample().detach().cpu().numpy()
            return Step(action=action)
        
    def timestep_complete(self, timestep: int, state, step: Step, reward, next_state, done) -> None:

        self.buffer.add(state, step.action, reward, next_state, done)

        if timestep > self.warmup_steps:

            if self.gradient_steps != -1 and timestep % self.update_freq == 0:
                for grad_update in range(self.gradient_steps):
                    self.learn()

            if self.gradient_steps == -1:
                self.learn()
        

    def get_stats(self) -> dict:
        return {}
