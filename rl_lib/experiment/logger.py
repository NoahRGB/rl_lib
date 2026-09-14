import os, pickle
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

tensorboard = True
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    tensorboard = False

from rl_lib.envs.env import EnvDetails

class Logger:
    def __init__(self, files_log_dir: str, tensorboard_log_dir: str,
                  use_tensorboard: bool, use_files: bool, print_progress: bool,
                  save_network: bool, title: str):

        self.title = title
        self.files_log_dir = files_log_dir
        self.tensorboard_log_dir = tensorboard_log_dir
        self.use_tensorboard = use_tensorboard and tensorboard
        self.use_files = use_files
        self.save_network = save_network
        self.print_progress = print_progress
        self.files_log_dir = self.files_log_dir
        self.tensorboard_log_dir = self._check_save_log(os.path.join(tensorboard_log_dir, self.title))

        if self.use_tensorboard:
            self.writer = SummaryWriter(self.tensorboard_log_dir )

    def _check_save_log(self, save_loc: str):
        if os.path.isdir(save_loc):
            current_digit = 1
            current_check = f"{save_loc}_{current_digit}"
            while os.path.isdir(current_check):
                current_digit += 1
                current_check = f"{save_loc}_{current_digit}"
            return current_check
        return save_loc

    def setup(self, env: EnvDetails):
        self.num_envs = env.num_envs
        self.running_rewards = np.zeros(self.num_envs)
        self.network_dict = {}
        self.timesteps_completed = 0
        self.stats = {"episodic_reward": [], "mean_episodic_reward": []}

    def _log_to_tensorboard(self):
        if self.use_tensorboard:
            for stat_name, stat_values in self.stats.items():
                if len(stat_values) > 0:
                    self.writer.add_scalar(stat_name, stat_values[-1], self.timesteps_completed)

    def _log_to_files(self):
        if self.use_files:
            for stat_name, stat_values in self.stats.items():
                with open(os.path.join(self.files_log_dir, f"{stat_name}.pkl"), "wb") as f:
                    pickle.dump(stat_values[0], f)
        if self.save_network:
            torch.save(self.network_dict, os.path.join(self.files_log_dir, "torch_checkpoint.pt"))

    def timestep_complete(self, rewards: np.array, dones, agent_stats: dict):
        if "metrics" in agent_stats:
            for stat_name, stat_value in agent_stats["metrics"].items():
                if stat_name not in self.stats:
                    self.stats[stat_name] = []
                self.stats[stat_name].append(stat_value)

        if "network" in agent_stats:
            self.network_dict = agent_stats["network"]

        self.timesteps_completed += self.num_envs
        self.running_rewards += rewards
        for env_idx, is_done in enumerate(dones):
            if is_done:
                episode_reward = self.running_rewards[env_idx]
                self.stats["episodic_reward"].append(episode_reward)
                self.stats["mean_episodic_reward"].append(np.mean(self.stats["episodic_reward"][-100:]))
                print(f"episode {len(self.stats['episodic_reward'])}, timesteps {self.timesteps_completed}, reward {episode_reward}")
                self.running_rewards[env_idx] = 0.0
                self._log_to_tensorboard()
                self._log_to_files()

    def training_done(self):
        ...