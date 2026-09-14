import torch

from rl_lib.envs.env import EnvDetails
from rl_lib.utils.heads import detect_head
from rl_lib.utils.encoders import detect_encoder


class ActorCriticNetwork(torch.nn.Module):

    def __init__(self, env: EnvDetails, lr: float, architecture: dict):
        super(ActorCriticNetwork, self).__init__()

        self.shared_enc = detect_encoder(architecture["shared_enc"], input_shape=env.state_space.shape)
        self.enc_output = self.shared_enc.encoder_output

        self.actor_head = detect_head(action_space=env.action_space, input_size=self.enc_output)
        self.value_head = torch.nn.Linear(self.enc_output, 1)

    def get_head_type(self):
        return type(self.actor_head)

    def forward(self, inp):
        enc_output = self.shared_enc(inp)
        actor_output = self.actor_head(enc_output)
        value_output = self.value_head(enc_output)
        
        return actor_output, value_output
        