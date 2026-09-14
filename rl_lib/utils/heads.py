import torch

from rl_lib.utils.spaces import Discrete, Continuous

def detect_head(action_space: Discrete|Continuous, input_size: int):

    if isinstance(action_space, Discrete):
        return CategoricalHead(input_size=input_size, output_size=action_space.n)
    
    elif isinstance(action_space, Continuous):
        return None

class CategoricalHead(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(CategoricalHead, self).__init__()

        self.head = torch.nn.Linear(input_size, output_size)

    def forward(self, inp):
        return self.head(inp)