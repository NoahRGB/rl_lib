import numpy as np
import torch

class Batch:

    def __init__(self, states: torch.tensor, actions: torch.tensor, rewards: torch.tensor, 
                 next_states: torch.tensor, dones: torch.tensor, log_probs: torch.tensor = None,
                 flattened: bool = False):
        
        # if not flattened, all of shape (tmax, num_envs, dim)
        # if flattened, all of shape (tmax*num_envs, dim)
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.log_probs = log_probs
        self.masks = 1 - self.dones
        self.is_flattened = flattened

    def gae(self, state_values: torch.tensor, final_state_values: torch.tensor, gamma: float, lam: float, device: torch.device):
        if not self.is_flattened:
            # cannot compute GAE on flattened batch, needs (tmax, num_envs) shape
            
            gae = 0.0
            tmax = self.rewards.shape[0]
            advantages = torch.zeros_like(self.rewards).to(device)
            state_values = state_values.squeeze(-1) # (tmax, num_envs)
            final_state_values = final_state_values.squeeze(-1)  # (num_envs,)
            next_value = final_state_values
            for t in reversed(range(tmax)):
                delta = self.rewards[t] + gamma * next_value * self.masks[t] - state_values[t]
                gae = delta + gamma * lam * self.masks[t] * gae
                advantages[t] = gae
                next_value = state_values[t]
            returns = advantages + state_values
            return advantages, returns
        
        return None, None

    def get_minibatch(self, indices: np.array):
        return Batch(self.states[indices], 
                     self.actions[indices], 
                     self.rewards[indices], 
                     self.next_states[indices], 
                     self.dones[indices], 
                     self.log_probs[indices] if self.log_probs is not None else None)

    def flatten(self):
        if not self.is_flattened:
            tmax, num_envs = self.states.shape[:2]
            return Batch(
                self.states.reshape(tmax*num_envs, *self.states.shape[2:]), 
                self.actions.reshape(tmax*num_envs, *self.actions.shape[2:]), 
                self.rewards.reshape(tmax*num_envs), 
                self.next_states.reshape(tmax*num_envs, *self.next_states.shape[2:]), 
                self.dones.reshape(tmax*num_envs), 
                self.log_probs.reshape(tmax*num_envs) if self.log_probs is not None else None,
                flattened=True
            )
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
        self.states_buffer[self.pointer] = states
        self.actions_buffer[self.pointer] = actions
        self.rewards_buffer[self.pointer] = rewards
        self.next_states_buffer[self.pointer] = next_states
        self.dones_buffer[self.pointer] = dones
        self.log_probs_buffer[self.pointer] = log_probs
        self.pointer += 1
        self.current_size += 1

    def get(self, device: torch.device) -> Batch:
        return Batch(torch.from_numpy(self.states_buffer).to(device), 
                     torch.from_numpy(self.actions_buffer).to(device), 
                     torch.from_numpy(self.rewards_buffer).to(device), 
                     torch.from_numpy(self.next_states_buffer).to(device), 
                     torch.from_numpy(self.dones_buffer).to(device), 
                     torch.from_numpy(self.log_probs_buffer).to(device))

    def clear(self):
        self.current_size = 0
        self.pointer = 0
