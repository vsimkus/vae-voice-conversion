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
        super(VAEGAN, self).__init__()
        raise NotImplementedError

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        raise NotImplementedError

class VQVAE(nn.Module):
    def __init__(self, input_shape, encoder_arch, vq_arch, generator_arch, num_speakers):
        super(VQVAE, self).__init__()
        self.input_shape = input_shape
        self.encoder_arch = encoder_arch
        self.vq_arch = vq_arch
        self.generator_arch = generator_arch
        self.num_speakers = num_speakers

        self.build_module()

    def build_module(self):
        print('Building VQVAE.')

        x = torch.zeros((self.input_shape))
        self.encoder = Encoder(input_shape=self.input_shape, 
                            kernel_sizes=self.encoder_arch.kernel_sizes,
                            strides=self.encoder_arch.strides,
                            num_residual_channels=self.encoder_arch.num_residual_channels,
                            latent_dim=self.vq_arch.latent_dim)
        
        x = self.encoder(x)

        self.vq = VectorQuantizer(embedding_dim=self.vq_arch.latent_dim, 
                                num_embeddings=self.vq_arch.num_latent)
        
        x_st, x = self.vq(x)

        self.generator = Generator(input_shape=x_st.shape,
                                num_speakers=self.num_speakers,
                                speaker_dim=self.generator_arch.speaker_dim,
                                kernel_sizes=self.generator_arch.kernel_sizes,
                                strides=self.generator_arch.strides,
                                out_paddings=self.generator_arch.out_paddings,
                                num_residual_channels=self.generator_arch.num_residual_channels)

        y = torch.zeros((self.input_shape[0]), dtype=torch.long)

        x_st = self.generator(x_st, y)    
    
    def forward(self, input, speaker):
        z = self.encoder(input)

        # Straight-through z is passed to decoder, the other is used for computing embedding update
        z_st, z_emb = self.vq(z)
        
        x_hat = self.generator(z_st, speaker)

        return x_hat, z_emb, z



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # TODO: use Encoder/Generator
        raise NotImplementedError

class Encoder(nn.Module):
    """
    Downsampling encoder with strided convolutions.
    """
    def __init__(self, input_shape, kernel_sizes, strides, num_residual_channels, latent_dim):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.num_residual_channels = num_residual_channels
        self.latent_dim = latent_dim

        self.layer_dict = nn.ModuleDict()
        self.build_module()
    
    def build_module(self):
        num_layers = len(self.kernel_sizes)
        print('Building Encoder with {} downsampling layers.'.format(num_layers))

        x = torch.zeros((self.input_shape))

        # Downsampling convolutions
        for i in range(num_layers):
            gated_conv = GatedConv1d(in_channels=x.shape[1],
                                    out_channels=self.num_residual_channels[i],
                                    kernel_size=self.kernel_sizes[i], 
                                    stride=self.strides[i],
                                    dilation=1)
            self.layer_dict['gated_conv_{}'.format(i)] = gated_conv

            # Trim the trailing pads
            # TODO: What should be the chomping when dilation is used?
            # chomp_conv = Chomp1d(int((self.kernel_size-1)/self.stride)) # TODO: confirm that the dim reduction is correct
            # self.layer_dict['chomp_conv_{}'.format(i)] = chomp_conv

            # x = chomp_conv(gated_conv(x))
            x = gated_conv(x)
        
        # Final convolution to output latents
        latent_conv = nn.Conv1d(in_channels=x.shape[1], 
                        out_channels=self.latent_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.layer_dict['latent_conv'] = latent_conv
    
    def forward(self, input):
        num_layers = len(self.kernel_sizes)
        out = input
        for i in range(num_layers):
            out = self.layer_dict['gated_conv_{}'.format(i)](out)
            # out = self.layer_dict['chomp_conv_{}'.format(i)](out)
        
        return self.layer_dict['latent_conv'](out)


class Generator(nn.Module): 
    """
    Generator or Decoder in the VAE using transposed 1-dimensional convolutions conditioned on the speaker id.
    """
    def __init__(self, input_shape, num_speakers, speaker_dim, kernel_sizes, strides, out_paddings, num_residual_channels):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.out_paddings = out_paddings
        self.num_residual_channels = num_residual_channels
        self.num_speakers = num_speakers
        self.speaker_dim = speaker_dim

        self.speaker_dict = nn.Embedding(num_embeddings=self.num_speakers,
                                        embedding_dim=self.speaker_dim)
                                        #TODO: init embeddings?
                                        # The original paper instead used one-hot encodings.
        self.layer_dict = nn.ModuleDict() 
        self.build_module()

    def build_module(self):
        num_layers = len(self.kernel_sizes)
        print('Building Decoder/Generator with {} upsampling layers.'.format(num_layers))
        
        x = torch.zeros((self.input_shape))
        y = torch.zeros((self.input_shape[0]), dtype=torch.long) # (batch) vector of speaker ids
        h = self.speaker_dict(y) # (batch, speaker_dim)
        
        # Upsampling convolutions
        for i in range(num_layers):
            conv = CondGatedTransposeConv1d(in_channels=x.shape[1],
                                    out_channels=self.num_residual_channels[i],
                                    cond_dim=self.speaker_dim,
                                    kernel_size=self.kernel_sizes[i], 
                                    stride=self.strides[i],
                                    out_padding=self.paddings[i],
                                    dilation=1)
            self.layer_dict['cond_gated_trans_conv_{}'.format(i)] = conv
            x = conv(x,h)
    
    def forward(self, input, speaker):
        num_layers = len(self.kernel_sizes)

        speaker_code = self.speaker_dict(speaker)
        out = input
        for i in range(num_layers):
            out = self.layer_dict['cond_gated_trans_conv_{}'.format(i)](out, speaker_code)
        
        return out


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initializes Vector Quantizer.
        """
        super(VectorQuantizer, self).__init__()
        print('Building VQ layer.')
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1./num_embeddings, 1./num_embeddings)

    def forward(self, input):
        """
        Forwards the input through the discrete layer and returns the mapped outputs.

        Returns:
            quantized_st    quantized input for straight-through gradient, such that the gradient only propagates to the inputs.
            quantized       quantized input that will propagate the gradient to the embedding vectors.
        """
        # Change to TF channel first order
        # (batch, channel, time) -> (batch, time, channel)
        input = input.permute(0, 2, 1).contiguous()

        # We prevent decoder gradients from reaching embeddings with weight.detach()
        # The gradients should still backpropagate to the inputs (st -- straight-through)
        quantized_st, latents = QuantizeVector.apply(input, self.embedding.weight.detach())
        # Change to PyTorch channel first order
        # (batch, time, channel) -> (batch, channel, time)
        quantized_st = quantized_st.permute(0, 2, 1).contiguous()

        # Another quantized output, that is specifically for propagating gradients to update embedding vectors.
        quantized = self.embedding(latents)
        quantized = quantized.permute(0, 2, 1).contiguous()

        return quantized_st, quantized

class GatedConv1d(nn.Module):
    """
    Gated convolution similar to WaveNet.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super(GatedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.build_module()
    
    def build_module(self):
        # TODO: padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
        # Only left-pads. no need for chomping afterwards.
        # from: https://github.com/NVIDIA/nv-wavenet/blob/master/pytorch/wavenet.py
        # padding = int((self.kernel_size-1) * self.dilation)
        padding = 0
        self.conv = nn.Conv1d(in_channels=self.in_channels, 
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=padding)

        self.gate = nn.Conv1d(in_channels=self.in_channels, 
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=padding)
    
    def forward(self, input):
        return torch.tanh(self.conv(input)) * torch.sigmoid(self.gate(input))

class CondGatedTransposeConv1d(nn.Module):
    """
    Conditional gated 1-dimensional convolution transpose.
    """
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size, stride, out_padding=0, dilation=1):
        super(CondGatedTransposeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_padding = out_padding
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.cond = nn.Linear(in_features=self.cond_dim,
                            out_features=self.out_channels)

        # TODO: padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
        # Only left-pads. no need for chomping afterwards.
        # from: https://github.com/NVIDIA/nv-wavenet/blob/master/pytorch/wavenet.py
        # padding = int((self.kernel_size-1) * self.dilation)
        # padding=0
        self.conv = nn.ConvTranspose1d(in_channels=self.in_channels, 
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        output_padding=self.padding)
        
        self.gate = nn.ConvTranspose1d(in_channels=self.in_channels, 
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        output_padding=self.padding)
        
    def forward(self, input, speaker):
        cond_out = self.cond(speaker).unsqueeze(-1)

        conv_out = self.conv(input)
        conv_out = torch.add(conv_out, cond_out)

        gate_out = self.gate(input)
        gate_out = torch.add(gate_out, cond_out)

        return torch.tanh(conv_out) * torch.sigmoid(gate_out)
        


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