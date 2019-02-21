import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import QuantizeVector


class VQVAEGAN(nn.Module):
    def __init__(self):
        raise NotImplementedError

class VAEGAN(nn.Module):
    def __init__(self):
        raise NotImplementedError

class Discriminator(nn.Module):
    def __init__(self):
        raise NotImplementedError

class VQVAE(nn.Module):
    def __init__(self):
        #TODO use Encoder/Generator/VectorQuantizer
        raise NotImplementedError

class VAE(nn.Module):
    def __init__(self):
        # TODO: use Encoder/Generator
        raise NotImplementedError

class Encoder(nn.Module):
    """
    Downsampling encoder with strided causal convolutions, similar to WaveNet.
    """
    def __init__(self, input_shape, num_layers, kernel_size, stride, num_residual_channels, latent_dim):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_residual_channels = num_residual_channels
        self.latent_dim = latent_dim

        self.layer_dict = nn.ModuleDict()
        self.build_module()
    
    def build_module(self):
        print('Building Encoder.')
        x = torch.zeros((self.input_shape))

        # Downsampling convolutions
        for i in range(self.num_layers):
            gated_conv = GatedConv1d(in_channels=x.shape[1],
                                    out_channels=(i+1)*self.num_residual_channels, # TODO: Check if this is not too much
                                    kernel_size=self.kernel_size, 
                                    stride=self.stride)
            self.layer_dict['gated_conv_{}'.format(i)] = gated_conv

            # Trim the trailing pads
            chomp_conv = Chomp1d(int((self.kernel_size-1)/self.stride)) # TODO: confirm that the dim reduction is correct
            self.layer_dict['chomp_conv_{}'.format(i)] = chomp_conv

            x = chomp_conv(gated_conv(x))
        
        # Final convolution to output latents
        latent_conv = nn.Conv1d(in_channels=x.shape[1], 
                        out_channels=self.latent_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.layer_dict['latent_conv'] = latent_conv
    
    def forward(self, input):
        out = input
        for i in range(self.num_layers):
            out = self.layer_dict['gated_conv_{}'.format(i)](out)
            out = self.layer_dict['chomp_conv_{}'.format(i)](out)
        
        return self.layer_dict['latent_conv'](out)


class Generator(nn.Module): 
    # Or we can name it Decoder. I've referred to it as the Generator in the report because it's
    # more convenient, the letter G doesn't clash with D for the discriminator :)
    def __init__(self):
        raise NotImplementedError

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initializes Vector Quantizer.
        """
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1./num_embeddings, 1./num_embeddings)

    def forward(self, input):
        """
        Forwards the input through the discrete layer and returns the mapped outputs.

        Returns:
            quantized_st    quantized input for straight-through gradient, such that the gradient only propagates to the inputs.
            quantized       quantized input that will propagate the gradient to the embedding vectors.
        """
        print('Building VQ layer.')
        # Change to TF channel first order
        # (batch, channel, width, height) -> (batch, width, height, channel)
        input = input.permute(0, 2, 3, 1).contiguous()

        # We prevent decoder gradients from reaching embeddings with weight.detach()
        # The gradients should still backpropagate to the inputs (st -- straight-through)
        quantized_st, latents = QuantizeVector.apply(input, self.embedding.weight.detach())
        # Change to PyTorch channel last order
        # (batch, width, height, channel) -> (batch, channel, width, height)
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()

        # Another quantized output, that is specifically for propagating gradients to update embedding vectors.
        quantized = self.embedding(latents)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized_st, quantized

class GatedConv1d(nn.Module):
    """
    Gated convolution similar to WaveNet used in the encoder.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GatedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.build_module()
    
    def build_module(self):
        self.conv = nn.Conv1d(in_channels=self.in_channels, 
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.kernel_size-1)

        self.gate = nn.Conv1d(in_channels=self.in_channels, 
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.kernel_size-1)
    
    def forward(self, input):
        return torch.tanh(self.conv(input)) * torch.sigmoid(self.gate(input))
        

class Chomp1d(nn.Module):
    """
    Module to cut the padding on one side of the tensor.
    Required for causal convolutions, such that current timestep does not depend on the future timestep

    Code borrowed from:
    https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class FCCNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a fully connected network similar to the ones implemented previously in the MLP package.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every fcc layer.
        :param num_layers: Number of fcc layers (excluding dim reduction stages)
        :param use_bias: Whether our fcc layers will use a bias.
        """
        super(FCCNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print("Building basic block of FCCNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            self.layer_dict['fcc_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # initialize a fcc layer
                                                            out_features=self.num_filters,
                                                            bias=self.use_bias)

            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        self.logits_linear_layer = nn.Linear(in_features=out.shape[1],  # initialize the prediction output linear layer
                                             out_features=self.num_output_classes,
                                             bias=self.use_bias)
        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, dim_reduction_type, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(ConvolutionalNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.dim_reduction_type = dim_reduction_type
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        print("Building basic block of ConvolutionalNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers

        out = x
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        for i in range(self.num_layers):  # for number of layers times
            self.layer_dict['conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                             # add a conv layer in the module dict
                                                             kernel_size=3,
                                                             out_channels=self.num_filters, padding=1,
                                                             bias=self.use_bias)

            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            out = F.relu(out)  # apply relu
            print(out.shape)
            if self.dim_reduction_type == 'strided_convolution':  # if dim reduction is strided conv, then add a strided conv
                self.layer_dict['dim_reduction_strided_conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                                                       kernel_size=3,
                                                                                       out_channels=out.shape[1],
                                                                                       padding=1,
                                                                                       bias=self.use_bias, stride=2,
                                                                                       dilation=1)

                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](
                    out)  # use strided conv to get an output
                out = F.relu(out)  # apply relu to the output
            elif self.dim_reduction_type == 'dilated_convolution':  # if dim reduction is dilated conv, then add a dilated conv, using an arbitrary dilation rate of i + 2 (so it gets smaller as we go, you can choose other dilation rates should you wish to do it.)
                self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                                                       kernel_size=3,
                                                                                       out_channels=out.shape[1],
                                                                                       padding=1,
                                                                                       bias=self.use_bias, stride=1,
                                                                                       dilation=i + 2)
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](
                    out)  # run dilated conv on input to get output
                out = F.relu(out)  # apply relu on output

            elif self.dim_reduction_type == 'max_pooling':
                self.layer_dict['dim_reduction_max_pool_{}'.format(i)] = nn.MaxPool2d(2, padding=1)
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)

            elif self.dim_reduction_type == 'avg_pooling':
                self.layer_dict['dim_reduction_avg_pool_{}'.format(i)] = nn.AvgPool2d(2, padding=1)
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)

            print(out.shape)
        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out,
                                        2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).
        print('shape before final linear layer', out.shape)
        out = out.view(out.shape[0], -1)
        self.logit_linear_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=self.use_bias)
        out = self.logit_linear_layer(out)  # apply linear layer on flattened inputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        for i in range(self.num_layers):  # for number of layers

            out = self.layer_dict['conv_{}'.format(i)](out)  # pass through conv layer indexed at i
            out = F.relu(out)  # pass conv outputs through ReLU
            if self.dim_reduction_type == 'strided_convolution':  # if strided convolution dim reduction then
                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](
                    out)  # pass previous outputs through a strided convolution indexed i
                out = F.relu(out)  # pass strided conv outputs through ReLU

            elif self.dim_reduction_type == 'dilated_convolution':
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](out)
                out = F.relu(out)

            elif self.dim_reduction_type == 'max_pooling':
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)

            elif self.dim_reduction_type == 'avg_pooling':
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)

        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.logit_linear_layer(out)  # pass through a linear layer to get logits/preds
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

        self.logit_linear_layer.reset_parameters()
