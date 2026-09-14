import numpy as np

from rl_lib.envs.env import EnvDetails

class Logger:
    def __init__(self, log_dir: str, use_tensorboard: bool, use_files: bool, print_progress: bool):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_files = use_files
        self.print_progress = print_progress

    def setup(self, env: EnvDetails):
        self.num_envs = env.num_envs
        self.episodic_rewards = []
        self.running_rewards = np.zeros(self.num_envs)

    def timestep_complete(self, rewards: np.array, dones, agent_stats: dict):
        self.running_rewards += rewards
        for env_idx, is_done in enumerate(dones):
            if is_done:
                episode_reward = self.running_rewards[env_idx]
                print(f"episode {len(self.episodic_rewards)+1}, reward {episode_reward}")
                self.episodic_rewards.append(episode_reward)
                self.running_rewards[env_idx] = 0.0

    def training_done(self):
        ...