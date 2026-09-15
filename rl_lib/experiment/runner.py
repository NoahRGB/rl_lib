import torch
import numpy as np
import random
import hydra

from rl_lib.algs.alg import Algorithm
from rl_lib.envs.env import Environment
from rl_lib.experiment.logger import Logger

def seeding(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def run_experiment(agent: Algorithm, env: Environment, logger: Logger, 
                   timesteps: int, seed: int, device: torch.device):
    seeding(seed)

    agent.setup(env.details, device)
    logger.setup(env.details)

    num_envs = env.details.num_envs
    timesteps_completed = 0
    states = env.get_start_states()


    while timesteps_completed < timesteps:
        
        step = agent.act(states)
        next_states, rewards, is_terms, is_truncs = env.step(step.action)
        dones = is_terms|is_truncs

        logger.timestep_complete(rewards, dones, agent.get_stats())
        agent.timestep_complete(timesteps_completed, states, step, rewards, next_states, is_terms)

        # gymnasium SyncVectorEnv reset modes
        # see https://farama.org/Vector-Autoreset-Mode
        
        # NEXT-STEP MODE:
        # env terminates at time t
        # so at time t+1 it resets()
        # so env is not ready for actions again until time t+2

        # THE FIX:
        # env terminates at time t
        # manually call a reset() at time t
        # so env is ready for actions again at time t+1
        states = env.reset(dones) if dones.any() else next_states

        timesteps_completed += num_envs
    
    logger.training_done()