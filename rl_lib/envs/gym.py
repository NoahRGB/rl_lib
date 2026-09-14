import gymnasium as gym

from rl_lib.envs.env import EnvDetails
from rl_lib.utils.spaces import Discrete, Continuous

def convert_gym_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        return Continuous(mins=space.low, maxs=space.high, shape=space.shape)

class GymEnv:
    
    def __init__(self, env_id: str, num_envs: int, seed: int = None) -> None:
        self.env = self._make_env(env_id, num_envs)
        self.seed = seed
        self.details = EnvDetails(num_envs=num_envs, 
                                  action_space=convert_gym_space(self.env.single_action_space), 
                                  state_space=convert_gym_space(self.env.single_observation_space))
        self.start_states, _ = self.reset()

    def _make_env(self, env_name: str, num_envs: int, **env_kwargs):

        def make_one_env():
            env = gym.make(env_name, **env_kwargs)
            return env
        
        try:
            env_list = [make_one_env for env_idx in range(num_envs)]
            env = gym.vector.SyncVectorEnv(env_list)
            env = gym.wrappers.vector.RecordEpisodeStatistics(env)
            return env

        except gym.error.NameNotFound as e:
            print(f"{env_name} not a valid Gymnasium environment")
            return None

    def reset(self):
        return self.env.reset(seed=self.seed)

    def get_start_states(self):
        return self.start_states

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return observation, reward, terminated, truncated
