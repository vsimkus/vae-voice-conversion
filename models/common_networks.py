import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_layers import LayerNormalizedGatedConv1d, LayerNormalizedLReLUConv1d, LayerNormalizedGatedTransposeConv1d, LayerNormalizedLReLUTransposeConv1d, Digital2Analog

class Encoder(nn.Module):
    """
    Downsampling encoder with strided convolutions.
    """
    def __init__(self, input_shape, kernel_sizes, strides, num_output_channels, 
                paddings, dilations, convolution_layer=LayerNormalizedGatedConv1d):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.num_output_channels = num_output_channels
        self.paddings = paddings
        self.dilations = dilations

        self.layer_dict = nn.ModuleDict()
        self.build_module(convolution_layer=convolution_layer)


    def build_module(self, convolution_layer):
        num_layers = len(self.kernel_sizes)
        print('Building Encoder with {} downsampling layers.'.format(num_layers))

        x = torch.zeros((self.input_shape))

        print(x.shape)
        # Downsampling convolutions
        for i in range(num_layers-1):
            conv = convolution_layer(input_shape=x.shape,
                                    in_channels=x.shape[1],
                                    out_channels=self.num_output_channels[i],
                                    kernel_size=self.kernel_sizes[i],
                                    stride=self.strides[i],
                                    padding=self.paddings[i],
                                    dilation=self.dilations[i])
            self.layer_dict['conv_{}'.format(i)] = conv

            x = conv(x)
            print(x.shape)
        
        self.final_conv = nn.Conv1d(in_channels=x.shape[1],
                                out_channels=self.num_output_channels[-1],
                                kernel_size=self.kernel_sizes[-1],
                                stride=self.strides[-1],
                                dilation=self.dilations[-1],
                                padding=self.paddings[-1])
        x = self.final_conv(x)
        print(x.shape)

    def forward(self, input):
        out = input
        for i in range(len(self.kernel_sizes)-1):
            out = self.layer_dict['conv_{}'.format(i)](out)

        out = self.final_conv(out)
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()
        
        self.final_conv.reset_parameters()


class Generator(nn.Module):
    """
    Generator or Decoder in the VAE using transposed 1-dimensional convolutions.
    """
    def __init__(self, input_shape, kernel_sizes, strides, dilations, paddings, 
                out_paddings, num_output_channels, convolution_layer=LayerNormalizedGatedTransposeConv1d):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilations = dilations
        self.paddings = paddings
        self.out_paddings = out_paddings
        self.num_output_channels = num_output_channels

        self.layer_dict = nn.ModuleDict()
        self.build_module(convolution_layer)

    def build_module(self, convolution_layer):
        num_layers = len(self.kernel_sizes)
        print('Building Decoder/Generator with {} upsampling layers.'.format(num_layers))

        x = torch.zeros((self.input_shape))

        print(x.shape)
        # Upsampling convolutions
        for i in range(num_layers-1):
            conv = convolution_layer(input_shape=x.shape,
                        in_channels=x.shape[1],
                        out_channels=self.num_output_channels[i],
                        kernel_size=self.kernel_sizes[i],
                        stride=self.strides[i],
                        dilation=self.dilations[i],
                        padding=self.paddings[i],
                        out_padding=self.out_paddings[i])
            self.layer_dict['conv_{}'.format(i)] = conv
            x = conv(x)
            print(x.shape)

        self.final_conv = nn.ConvTranspose1d(in_channels=x.shape[1],
                                    out_channels=self.num_output_channels[-1],
                                    kernel_size=self.kernel_sizes[-1],
                                    stride=self.strides[-1],
                                    dilation=self.dilations[-1],
                                    padding=self.paddings[-1],
                                    output_padding=self.out_paddings[-1])
        x = self.final_conv(x)
        print(x.shape)

    def forward(self, input):
        num_layers = len(self.kernel_sizes)

        out = input
        for i in range(num_layers-1):
            out = self.layer_dict['conv_{}'.format(i)](out)

        out = self.final_conv(out)

        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()
        
        self.final_conv.reset_parameters()

class QuantisedInputModuleWrapper(nn.Module):
    """
    Wrapper for any module that should take quantised (mu-law encoded) inputs
    """
    def __init__(self, num_input_quantization_channels, model):
        super(QuantisedInputModuleWrapper, self).__init__()
        print('Building Quantised input module.')
        self.d2a = Digital2Analog(num_input_quantization_channels)
        self.model = model
    
    def forward(self, digital_input, speaker):
        analog_input = self.d2a(digital_input)
        return self.model(analog_input, speaker)
    
    def reset_parameters(self):
        pass