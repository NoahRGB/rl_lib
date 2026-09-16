import gymnasium as gym
import ale_py

from rl_lib.envs.env import EnvDetails
from rl_lib.utils.spaces import Discrete, Continuous

def convert_gym_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n, shape=tuple())
    elif isinstance(space, gym.spaces.Box):
        return Continuous(mins=space.low, maxs=space.high, shape=space.shape)

gym.register(
    id="ThrustEnv",
    entry_point="src.thrust_gym.thrust_gymenv:ThrustEnv",
)

class GymEnv:
    
    def __init__(self, env_id: str, num_envs: int, is_atari: bool = False, 
                 normalise_obs: bool = False, seed: int = None, **env_kwargs) -> None:
        
        self.seed = seed
        self.has_reset = False # has the env been reset yet?
        self.is_atari = is_atari
        self.normalise_obs = normalise_obs
        self.env = self._make_env(env_id, num_envs, **env_kwargs)
        if self.env is None:
            raise ValueError(f"Environment {env_id} could not be created")

        self.details = EnvDetails(num_envs=num_envs, 
                                  action_space=convert_gym_space(self.env.single_action_space), 
                                  state_space=convert_gym_space(self.env.single_observation_space))
        self.start_states = self.reset()

    def _make_env(self, env_name: str, num_envs: int, **env_kwargs):

        def make_one_env():
            env = gym.make(env_name, **env_kwargs)

            if self.is_atari:
                env = gym.make(env_name, frameskip=1, **env_kwargs)
                env = gym.wrappers.AtariPreprocessing(env,
                    noop_max=30, frame_skip=4, terminal_on_life_loss=False,
                    screen_size=84, grayscale_obs=True, grayscale_newaxis=False
                )
                env = gym.wrappers.FrameStackObservation(env, stack_size=4)
                # env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
                # env = gym.wrappers.ResizeObservation(env, (84, 84))
                # env = gym.wrappers.FrameStackObservation(env, stack_size=4)

            if self.normalise_obs:
                env = gym.wrappers.NormalizeObservation(env)
                
            return env
        
        try:
            env_list = [make_one_env for env_idx in range(num_envs)]
            env = gym.vector.SyncVectorEnv(env_list)
            env = gym.wrappers.vector.RecordEpisodeStatistics(env)
            return env

        except gym.error.NameNotFound as e:
            print(f"{env_name} not a valid Gymnasium environment")
            return None

    def reset(self, reset_mask=None):
        # only use self.seed for the first reset
        seed = self.seed if not self.has_reset else None
        self.has_reset = True

        # only reset sub envs from reset_mask
        options = {"reset_mask": reset_mask} if reset_mask is not None else None
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs

    def get_start_states(self):
        return self.start_states

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return observation, reward, terminated, truncated
