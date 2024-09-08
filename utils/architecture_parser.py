import torch.nn as nn
from utils.utils import conv2d_output_shape, deconv2d_output_shape
import numpy as np

def parse_architecture(architecture, input_shape=None):
    seq = []
    layer_shapes = [input_shape]
    layers = architecture['layers']
    for a in layers:
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
            if input_shape is not None:
                layer_shapes.append(
                    conv2d_output_shape(
                        input_shape=layer_shapes[-1],
                        kernel_size=a_params['kernel_size'],
                        stride=a_params['stride'],
                        padding=a_params['padding']
                    )
                )
        elif a['function'] == 'deconv2d':
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
            if input_shape is not None:
                layer_shapes.append(
                    deconv2d_output_shape(
                        input_shape=layer_shapes[-1],
                        kernel_size=a_params['kernel_size'],
                        stride=a_params['stride'],
                        padding=a_params['padding'],
                        output_padding=0
                    )
                )
        elif a['function'] == 'lstm':
            # Add functional layer
            action = nn.LSTM(
                input_size=int(a_params['input_size']),
                hidden_size=int(a_params['hidden_size']),
                num_layers=int(a_params['num_layers']) ,
                bidirectional=True,
                batch_first=True
            )

            # # keep track of the convolution shapes
            # layer_shapes.append(
            #     deconv2d_output_shape(
            #         input_shape=layer_shapes[-1],
            #         kernel_size=a_params['kernel_size'],
            #         stride=a_params['stride'],
            #         padding=a_params['padding'],
            #         output_padding=0
            #     )
            # )
        elif a['function'] == 'batchNorm2d':
            action = nn.BatchNorm2d(a_params['size'])

        elif a['function'] == 'leakyrelu':
            action = nn.LeakyReLU(negative_slope=a_params['negative_slope'])
        elif a['function'] == 'sigmoid':
            action = nn.Sigmoid()

        seq.append(action)
    return seq, layer_shapes
