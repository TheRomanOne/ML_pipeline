import torch.nn as nn
from utils.utils import conv2d_output_shape, deconv2d_output_shape
import numpy as np
def parse_architecture(input_shape, architecture):
    seq = []
    layer_shapes = [input_shape]
    for a in architecture:
        a_params = a['params']
        if a['function'] == 'conv2d':
            # Add functional layer
            action = nn.Conv2d(
                in_channels=a_params['in_channels'],
                out_channels=a_params['out_channels'],
                kernel_size=a_params['kernel_size'],
                stride=a_params['stride'],
                padding=a_params['padding']
            )

            # keep track of the convolution shapes
            layer_shapes.append(
                conv2d_output_shape(
                    input_shape=layer_shapes[-1],
                    kernel_size=a_params['kernel_size'],
                    stride=a_params['stride'],
                    padding=a_params['padding']
                )
            )
        if a['function'] == 'deconv2d':
            # Add functional layer

            # TODO: remove hack
            in_channels = a_params['in_channels'] if a_params['in_channels'] > 0 else np.max(input_shape) * 2

            action = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=a_params['out_channels'],
                kernel_size=a_params['kernel_size'],
                stride=a_params['stride'],
                padding=a_params['padding']
            )

            # keep track of the convolution shapes
            layer_shapes.append(
                deconv2d_output_shape(
                    input_shape=layer_shapes[-1],
                    kernel_size=a_params['kernel_size'],
                    stride=a_params['stride'],
                    padding=a_params['padding'],
                    output_padding=0
                )
            )
        elif a['function'] == 'batchNorm2d':
            action = nn.BatchNorm2d(a_params['size'])

        elif a['function'] == 'relu':
            action = nn.LeakyReLU(negative_slope=a_params['negative_slope'])

        seq.append(action)
    return seq, layer_shapes
