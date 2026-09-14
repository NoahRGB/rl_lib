import numpy as np
import torch

class Batch:

    def __init__(self, states: torch.tensor, actions: torch.tensor, rewards: torch.tensor, next_states: torch.tensor, dones: torch.tensor, log_probs: torch.tensor = None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.log_probs = log_probs

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

    def is_full(self):
        return self.current_size >= self.max_size

    def add(self, states: np.array, actions: np.array, rewards: np.array, next_states: np.array, dones: np.array, log_probs: np.array=None):
        self.states[self.pointer] = states
        self.actions_buffer[self.pointer] = actions
        self.rewards_buffer[self.pointer] = rewards
        self.next_states_buffer[self.pointer] = next_states
        self.dones_buffer[self.pointer] = dones
        self.log_probs_buffer[self.pointer] = log_probs
        self.pointer += 1

    def get(self):
        return Batch(self.states, self.actions, self.rewards, self.next_states_buffer, self.dones, self.log_probs_buffer)

    def clear(self):
        self.current_size = 0
        self.pointer = 0
