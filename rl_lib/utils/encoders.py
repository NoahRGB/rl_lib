import torch

from rl_lib.utils.network_builder import build_network

def detect_encoder(enc: dict, input_shape: tuple = None, output_shape: tuple = None):
    if enc["type"] == "MLP":
        return MLPEnc(architecture=enc["layers"], 
                      input_shape=input_shape, 
                      output_shape=output_shape)
    elif enc["type"] == "CNN":
        return None
    else:
        print(f"Encoder type {enc['type']} not supported")
        return None

class MLPEnc(torch.nn.Module):

    def __init__(self, architecture: list, input_shape: tuple = None, output_shape: tuple = None):
        super(MLPEnc, self).__init__()

        self.body, self.encoder_output = build_network(architecture=architecture, 
                                                       input_shape=input_shape, 
                                                       output_shape=output_shape)

    def forward(self, inp):
        return self.body(inp)


class CNNEnc:

    def __init__(self):
        ...