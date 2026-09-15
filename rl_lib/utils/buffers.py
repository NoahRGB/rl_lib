import numpy as np
import torch

class Batch:
    def __init__(self, states: torch.tensor, actions: torch.tensor, rewards: torch.tensor, 
                 next_states: torch.tensor, dones: torch.tensor, log_probs: torch.tensor = None,
                 intrinsic_rewards: torch.tensor = None):

        # tensors should all be of shape (tmax, batch, *dim)
        # batch could be the num_envs, the minibatch_size, etc.

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.log_probs = log_probs
        self.intrinsic_rewards = intrinsic_rewards
        self.masks = 1 - self.dones

    def gae(self, state_values: torch.tensor, final_state_values: torch.tensor, gamma: float, lam: float, device: torch.device):
        if len(self.states.shape) > 2:
            # cannot compute GAE on flattened batch, needs (tmax, num_envs) shape
            
            gae = 0.0
            tmax = self.rewards.shape[0]
            advantages = torch.zeros_like(self.rewards).to(device)
            next_value = final_state_values
            for t in reversed(range(tmax)):
                delta = self.rewards[t] + gamma * next_value * self.masks[t] - state_values[t]
                gae = delta + gamma * lam * self.masks[t] * gae
                advantages[t] = gae
                next_value = state_values[t]
            returns = advantages + state_values
            return advantages, returns
        
        return None, None

    def nstep_returns(self, last_values: torch.tensor, gamma: float, device: torch.device):
        if len(self.states.shape) > 2:
            tmax, batch = self.states.shape[:2]
            returns = torch.zeros_like(self.rewards).to(device)
            next_value = last_values
            for t in reversed(range(tmax)):
                returns[t] = self.rewards[t] + gamma * next_value * self.masks[t]
                next_value = returns[t]
            return returns
        return None

    def get_minibatch(self, indices: np.array):
        return Batch(self.states[indices], 
                     self.actions[indices], 
                     self.rewards[indices], 
                     self.next_states[indices], 
                     self.dones[indices], 
                     self.log_probs[indices] if self.log_probs is not None else None,
                     self.intrinsic_rewards[indices] if self.intrinsic_rewards is not None else None)

    def flatten(self):
        if len(self.states.shape) > 2:
            tmax, batch = self.states.shape[:2]
            return Batch(
                self.states.reshape(tmax*batch, *self.states.shape[2:]), 
                self.actions.reshape(tmax*batch, *self.actions.shape[2:]), 
                self.rewards.reshape(tmax*batch), 
                self.next_states.reshape(tmax*batch, *self.next_states.shape[2:]), 
                self.dones.reshape(tmax*batch), 
                self.log_probs.reshape(tmax*batch) if self.log_probs is not None else None,
                self.intrinsic_rewards.reshape(tmax*batch) if self.intrinsic_rewards is not None else None
            )
        else:
            return None

class OffPolicyBuffer:

    def __init__(self, size: int, num_envs: int, state_dim: tuple, action_dim: tuple):
        self.max_size = size
        self.num_envs = num_envs
        self.current_size = 0
        self.pointer = 0

        self.states_buffer = np.empty((self.max_size, *state_dim), dtype=np.float32)
        self.actions_buffer = np.empty((self.max_size, *action_dim), dtype=np.float32)
        self.rewards_buffer = np.empty((self.max_size), dtype=np.float32)
        self.next_states_buffer = np.empty((self.max_size, *state_dim), dtype=np.float32)
        self.dones_buffer = np.empty((self.max_size), dtype=np.float32)

    def add(self, states: np.array, actions: np.array, rewards: np.array, next_states: np.array, dones: np.array):

        # the (num_envs, ) structure is not retained
        # everything is flattened into single transitions
        for env in range(self.num_envs):
            self.states_buffer[self.pointer] = states[env]
            self.actions_buffer[self.pointer] = actions[env]
            self.next_states_buffer[self.pointer] = next_states[env]
            self.rewards_buffer[self.pointer] = rewards[env]
            self.dones_buffer[self.pointer] = dones[env]

            self.pointer = (self.pointer + 1) % self.max_size
            self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, size: int, device: torch.device) -> Batch:
        sample_size = min(size, self.current_size)
        sample_indices = np.random.choice(self.current_size, sample_size, replace=False)

        return Batch(
            torch.from_numpy(self.states_buffer[sample_indices]).unsqueeze(0).to(device),
            torch.from_numpy(self.actions_buffer[sample_indices]).unsqueeze(0).to(device),
            torch.from_numpy(self.rewards_buffer[sample_indices]).unsqueeze(0).to(device),
            torch.from_numpy(self.next_states_buffer[sample_indices]).unsqueeze(0).to(device),
            torch.from_numpy(self.dones_buffer[sample_indices]).unsqueeze(0).to(device)
        ), sample_size

class OnPolicyBuffer:

    def __init__(self, size: int, num_envs: int, state_dim: tuple, action_dim: tuple):
        self.max_size = size
        self.num_envs = num_envs
        self.current_size = 0
        self.pointer = 0

        self.states_buffer = np.empty((self.max_size, self.num_envs, *state_dim), dtype=np.float32)
        self.actions_buffer = np.empty((self.max_size, self.num_envs, *action_dim), dtype=np.float32)
        self.rewards_buffer = np.empty((self.max_size, self.num_envs), dtype=np.float32)
        self.next_states_buffer = np.empty((self.max_size, self.num_envs, *state_dim), dtype=np.float32)
        self.dones_buffer = np.empty((self.max_size, self.num_envs), dtype=np.float32)

        self.log_probs_buffer = np.empty((self.max_size, self.num_envs), dtype=np.float32)
        self.intrinsic_rewards = np.empty((self.max_size, self.num_envs), dtype=np.float32)

    def is_full(self):
        return self.current_size >= self.max_size

    def add(self, states: np.array, actions: np.array, rewards: np.array, 
            next_states: np.array, dones: np.array, log_probs: np.array=None,
            intrinsic_rewards: np.array=None):
        
        self.states_buffer[self.pointer] = states
        self.actions_buffer[self.pointer] = actions
        self.rewards_buffer[self.pointer] = rewards
        self.next_states_buffer[self.pointer] = next_states
        self.dones_buffer[self.pointer] = dones

        if log_probs is not None:
            self.log_probs_buffer[self.pointer] = log_probs
        if intrinsic_rewards is not None:
            self.intrinsic_rewards[self.pointer] = intrinsic_rewards

        self.pointer += 1
        self.current_size += 1

    def get(self, device: torch.device) -> Batch:
        return Batch(torch.from_numpy(self.states_buffer).to(device), 
                     torch.from_numpy(self.actions_buffer).to(device), 
                     torch.from_numpy(self.rewards_buffer).to(device), 
                     torch.from_numpy(self.next_states_buffer).to(device), 
                     torch.from_numpy(self.dones_buffer).to(device), 
                     torch.from_numpy(self.log_probs_buffer).to(device),
                     torch.from_numpy(self.intrinsic_rewards).to(device))

    def clear(self):
        self.current_size = 0
        self.pointer = 0
