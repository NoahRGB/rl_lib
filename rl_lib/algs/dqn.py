import torch
import numpy as np

from rl_lib.algs.alg import Step
from rl_lib.envs.env import EnvDetails
from rl_lib.utils.schedulers import detect_scheduler
from rl_lib.utils.networks import QNetwork
from rl_lib.utils.buffers import OffPolicyBuffer
from rl_lib.utils.spaces import Discrete

class DQN:

    def __init__(self, lr: float, replay_size: int, target_update_interval: int,
                 update_freq: int, gradient_steps: int, minibatch_size: int,
                 gamma: float, epsilon_scheduler: dict, cgn: float, warmup_steps: int,
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
        self.load_path = load_path

        self.epsilon_scheduler = detect_scheduler(epsilon_scheduler)
        self.epsilon = self.epsilon_scheduler.get_value()

    def setup(self, env: EnvDetails, device: torch.device) -> None:
        self.env = env
        self.device = device
        self.stats = {}
        assert type(self.env.action_space) == Discrete and self.env.num_envs == 1

        self.network = QNetwork(env, self.net_architecture).to(self.device)
        self.target_network = QNetwork(env, self.net_architecture).to(self.device)
        self._update_target_net()
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.buffer = OffPolicyBuffer(self.replay_size, env.num_envs, env.state_space.shape, env.action_space.shape)

    def _update_target_net(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def learn(self):
        full_batch, batch_size = self.buffer.sample(self.minibatch_size, self.device)
    
        if batch_size == self.minibatch_size:

            qvals = self.network(full_batch.states)
            chosen_qvals = qvals.gather(-1, full_batch.actions.long().unsqueeze(-1)).squeeze(-1)
            chosen_qvals = chosen_qvals.view(self.minibatch_size)

            with torch.no_grad():
                next_qvals = self.target_network(full_batch.next_states)
                targets = full_batch.nstep_returns(next_qvals.max(-1)[0], self.gamma, self.device)
                targets = targets.view(self.minibatch_size)

            loss = torch.nn.functional.mse_loss(chosen_qvals, targets)
            self.optim.zero_grad()
            loss.backward()
            if self.cgn is not None:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cgn)
            self.optim.step()

    def act(self, state) -> Step:
        with torch.no_grad():
            if np.random.random() >= self.epsilon:
                q_values = self.network(torch.from_numpy(state).to(self.device))
                action = q_values.argmax(dim=-1).cpu().numpy()
            else:
                action = np.array([np.random.randint(0, self.env.action_space.n, dtype=np.int64)])
            return Step(action=action)
        
    def timestep_complete(self, timestep: int, state, step: Step, reward, next_state, done) -> None:

        if timestep % self.target_update_interval == 0:
            self._update_target_net()


        self.buffer.add(state, step.action, reward, next_state, done)

        if timestep > self.warmup_steps:
            self.epsilon = self.epsilon_scheduler.step()

            if self.gradient_steps != -1 and timestep % self.update_freq == 0:
                for grad_update in range(self.gradient_steps):
                    self.learn()

            if self.gradient_steps == -1:
                self.learn()
        

    def get_stats(self) -> dict:
        return {}
