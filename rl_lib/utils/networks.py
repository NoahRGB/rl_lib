import torch

from rl_lib.envs.env import EnvDetails
from rl_lib.utils.heads import detect_head
from rl_lib.utils.encoders import detect_encoder
from rl_lib.utils.spaces import Discrete

class QNetwork(torch.nn.Module):

    def __init__(self, env: EnvDetails, architecture: dict):
        super(QNetwork, self).__init__()
        assert type(env.action_space) == Discrete

        self.encoder = detect_encoder(architecture["enc"], input_shape=env.state_space.shape)
        self.enc_out = self.encoder.encoder_output

        self.qvals_out = torch.nn.Linear(self.enc_out, env.action_space.n)

    def forward(self, inp):
        enc_out = self.encoder(inp)
        return self.qvals_out(enc_out)

class ActorCriticNetwork(torch.nn.Module):

    def __init__(self, env: EnvDetails, architecture: dict):
        super(ActorCriticNetwork, self).__init__()

        if "shared_enc" in architecture:
            self.using_shared_enc = True
            self.shared_enc = detect_encoder(architecture["shared_enc"], input_shape=env.state_space.shape)
            self.actor_enc_output = self.shared_enc.encoder_output
            self.value_enc_output = self.shared_enc.encoder_output

        else:
            self.using_shared_enc = False
            self.actor_enc = detect_encoder(architecture["actor_enc"], input_shape=env.state_space.shape)
            self.value_enc = detect_encoder(architecture["value_enc"], input_shape=env.state_space.shape)
            self.actor_enc_output = self.actor_enc.encoder_output
            self.value_enc_output = self.value_enc.encoder_output


        self.actor_head = detect_head(action_space=env.action_space, input_size=self.actor_enc_output)
        self.value_head = torch.nn.Linear(self.value_enc_output, 1)

    def get_head_type(self):
        return type(self.actor_head)

    def forward(self, inp):
        if not self.using_shared_enc:
            actor_enc_output = self.actor_enc(inp)
            value_enc_output = self.value_enc(inp)
            actor_output = self.actor_head(actor_enc_output)
            value_output = self.value_head(value_enc_output)
        else:
            enc_output = self.shared_enc(inp)
            actor_output = self.actor_head(enc_output)
            value_output = self.value_head(enc_output)
        
        return actor_output, value_output
        