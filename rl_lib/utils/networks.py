import torch

from rl_lib.envs.env import EnvDetails
from rl_lib.utils.heads import detect_head
from rl_lib.utils.encoders import detect_encoder
from rl_lib.utils.spaces import Discrete

class ICM_ActorCriticNetwork(torch.nn.Module):

    def __init__(self, env: EnvDetails, architecture: dict):
        super(ICM_ActorCriticNetwork, self).__init__()

        self.icm_enc = detect_encoder(architecture["icm_enc"], input_shape=env.state_space.shape)
        self.icm_enc_output = self.icm_enc.encoder_output

        self.icm_inv_model = detect_head(action_space=env.action_space, input_size=self.icm_enc_output*2)

        icm_forward_input_size = self.icm_enc_output+(1 if isinstance(env.action_space, Discrete) else env.action_space.shape[0])
        self.icm_forward_model = torch.nn.Linear(icm_forward_input_size, self.icm_enc_output)

        if "shared_ppo_enc" in architecture:
            self.using_shared_enc = True
            self.shared_enc = detect_encoder(architecture["shared_ppo_enc"], input_shape=env.state_space.shape)
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

    def actor_critic_out(self, state):
        if not self.using_shared_enc:
            actor_enc_output = self.actor_enc(state)
            value_enc_output = self.value_enc(state)
            actor_output = self.actor_head(actor_enc_output)
            value_output = self.value_head(value_enc_output)
        else:
            enc_output = self.shared_enc(state)
            actor_output = self.actor_head(enc_output)
            value_output = self.value_head(enc_output)
        
        return actor_output, value_output

    def inv_model(self, state, next_state):
        phi_state = self.icm_enc(state)
        phi_next_state = self.icm_enc(next_state)

        icm_inv_input = torch.concat([phi_state, phi_next_state], dim=-1)
        action_pred = self.icm_inv_model(icm_inv_input)

        return action_pred

    def forward_model(self, state, action, next_state):
        phi_state = self.icm_enc(state)
        phi_next_state = self.icm_enc(next_state)

        icm_forward_input = torch.concat([phi_state, action.unsqueeze(1)], dim=1)
        phi_next_state_pred = self.icm_forward_model(icm_forward_input)

        return phi_next_state, phi_next_state_pred
    

class QNetwork(torch.nn.Module):

    def __init__(self, env: EnvDetails, architecture: dict, input_shape=None, output_shape=None):
        super(QNetwork, self).__init__()

        self.encoder = detect_encoder(architecture["qnet_enc"], input_shape=env.state_space.shape if input_shape is None else input_shape)
        self.enc_out = self.encoder.encoder_output

        self.qvals_out = torch.nn.Linear(self.enc_out, env.action_space.n if isinstance(env.action_space, Discrete) else env.action_space.shape[0])

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

class ActorNetwork(torch.nn.Module):

    def __init__(self, env: EnvDetails, architecture: dict, is_deterministic: bool = False):
        super(ActorNetwork, self).__init__()
        self.is_deterministic = is_deterministic

        self.enc = detect_encoder(architecture["actor_enc"], input_shape=env.state_space.shape)
        self.enc_output = self.enc.encoder_output

        if not self.is_deterministic:
            self.actor_head = detect_head(action_space=env.action_space, input_size=self.enc_output)

    def get_head_type(self):
        if not self.is_deterministic:
            return type(self.actor_head)
        return None

    def forward(self, inp):
        enc_output = self.enc(inp)

        if self.is_deterministic:
            return enc_output
        
        actor_output = self.actor_head(enc_output)
        return actor_output
        