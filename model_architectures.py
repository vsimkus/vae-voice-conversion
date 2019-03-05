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

class GAN(nn.Module):
    def __init__(self, input_shape, discriminator_arch, generator_arch, num_speakers):
        super(GAN, self).__init__()
        self.input_shape = input_shape
        self.discriminator_arch = discriminator_arch
        self.generator_arch = generator_arch
        self.num_speakers = num_speakers

        self.build_module()

    def build_module(self):
        print("Building GAN.")

        #x = torch.zeros((self.input_shape))

        self.generator = Generator(input_shape=[1, self.generator_arch.latent_dim, self.generator_arch.latent_length],
                            num_speakers=self.num_speakers,
                            speaker_dim=self.generator_arch.speaker_dim,
                            kernel_sizes=self.generator_arch.kernel_sizes,
                            strides=self.generator_arch.strides,
                            dilations=self.generator_arch.dilations,
                            paddings=self.generator_arch.paddings,
                            out_paddings=self.generator_arch.out_paddings,
                            num_residual_channels=self.generator_arch.num_residual_channels)

        self.discriminator = Discriminator(input_shape=self.input_shape,
                                            kernel_sizes=self.discriminator_arch.kernel_sizes,
                                            strides=self.discriminator_arch.strides,
                                            num_residual_channels=self.discriminator_arch.num_residual_channels)

        #distr = normal.Normal(0.0, 1.0)
        #z = distr.sample(self.generator.input_shape)
        #h = self.generator.speaker_dict(torch.rand(1) * self.num_speakers)

    def forward(self, input, speaker, generator_latent):

        x_hat = self.generator(generator_latent, speaker)
        real_pred = self.discriminator(input)
        fake_pred = self.discriminator(x_hat)

        return x_hat, real_pred, fake_pred

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        # for item in self.layer_dict.children():
        self.generator.reset_parameters()
        self.discriminator.reset_parameters()

class Discriminator(nn.Module):
    def __init__(self, input_shape, kernel_sizes, strides, num_residual_channels):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.num_residual_channels = num_residual_channels

        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        num_layers = len(self.kernel_sizes)

        x = torch.zeros((self.input_shape))
        for i in range(num_layers):
            gated_conv = GatedConv1d(in_channels=x.shape[1],
                                    out_channels=self.num_residual_channels[i],
                                    kernel_size=self.kernel_sizes[i],
                                    stride=self.strides[i],
                                    dilation=1)
            self.layer_dict['gated_conv_{}'.format(i)] = gated_conv

            x = gated_conv(x)

            print(x.shape)

        final_conv = nn.Conv1d(in_channels=x.shape[1],
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0)

        self.layer_dict['final_conv'] = final_conv
        x = final_conv(x)
        print(x.shape)

        fully_connected = nn.Linear(in_features=x.shape[2], out_features=1)
        self.layer_dict['fully_connected'] = fully_connected
        x = fully_connected(x)
        print(x.shape)

        sigmoid = nn.Sigmoid()
        self.layer_dict['sigmoid'] = sigmoid
        x = sigmoid(x)
        print(x.shape)

    def forward(self, input):
        num_layers = len(self.kernel_sizes)
        out = input
        for i in range(num_layers):
            out = self.layer_dict['gated_conv_{}'.format(i)](out)

        out = self.layer_dict['fully_connected'](out)
        out = self.layer_dict['sigmoid'](out)
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        # Reset parameters for all layers except final sigmoid layer
        for i, item in enumerate(self.layer_dict.children()):
            print(item)
            if i != len(self.layer_dict) - 1:
                item.reset_parameters()
    
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
                                dilations=self.generator_arch.dilations,
                                paddings=self.generator_arch.paddings,
                                out_paddings=self.generator_arch.out_paddings,
                                num_residual_channels=self.generator_arch.num_residual_channels,
                                pre_output_channels=self.generator_arch.pre_output_channels, 
                                num_final_output_channels=self.generator_arch.num_final_output_channels)

        y = torch.zeros((self.input_shape[0]), dtype=torch.long)

        x_st = self.generator(x_st, y)    
    
    def forward(self, input, speaker):
        z = self.encoder(input)

        # Straight-through z is passed to decoder, the other is used for computing embedding update
        z_st, z_emb = self.vq(z)

        x_hat = self.generator(z_st, speaker)

        return x_hat, z, z_emb
    
    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        # for item in self.layer_dict.children():
        self.encoder.reset_parameters()
        self.vq.reset_parameters()
        self.generator.reset_parameters()

class VQVAEQuantizedInput(VQVAE):
    """
    VQVAE with input quantization, so we can use CrossEntropy loss.
    """
    def __init__(self, input_shape, encoder_arch, vq_arch, generator_arch, num_speakers, num_input_quantization_channels):
        super(VQVAEQuantizedInput, self).__init__(input_shape, encoder_arch, vq_arch, generator_arch, num_speakers)

        self.d2a = Digital2Analog(num_input_quantization_channels)
    
    def forward(self, input, speaker):
        analog_input = self.d2a(input)
        return super(VQVAEQuantizedInput, self).forward(analog_input, speaker)

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

        print(x.shape)
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
            print(x.shape)
        
        # Final convolution to output latents
        latent_conv = nn.Conv1d(in_channels=x.shape[1], 
                        out_channels=self.latent_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.layer_dict['latent_conv'] = latent_conv
        self.layer_dict['latent_batch_norm'] = nn.BatchNorm1d(self.latent_dim)
        
        x = latent_conv(x)
        print(x.shape)
    
    def forward(self, input):
        out = input
        for i in range(len(self.kernel_sizes)):
            out = self.layer_dict['gated_conv_{}'.format(i)](out)
            # out = self.layer_dict['chomp_conv_{}'.format(i)](out)
        
        out = self.layer_dict['latent_conv'](out)
        return self.layer_dict['latent_batch_norm'](out)
    
    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()


class Generator(nn.Module): 
    """
    Generator or Decoder in the VAE using transposed 1-dimensional convolutions conditioned on the speaker id.
    """
    def __init__(self, input_shape, num_speakers, 
                speaker_dim, kernel_sizes, strides, dilations, 
                paddings, out_paddings, num_residual_channels, 
                pre_output_channels, num_final_output_channels):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilations = dilations
        self.paddings = paddings
        self.out_paddings = out_paddings
        self.num_residual_channels = num_residual_channels
        self.pre_output_channels = pre_output_channels
        self.num_final_output_channels = num_final_output_channels
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
        y = torch.zeros((self.input_shape[0]), dtype=torch.long) # (batch,) vector of speaker ids
        h = self.speaker_dict(y) # (batch, speaker_dim)
        
        print(x.shape)
        # Upsampling convolutions
        for i in range(num_layers):
            conv = CondGatedTransposeConv1d(in_channels=x.shape[1],
                                    out_channels=self.num_residual_channels[i],
                                    cond_dim=self.speaker_dim,
                                    kernel_size=self.kernel_sizes[i], 
                                    stride=self.strides[i],
                                    dilation=self.dilations[i],
                                    padding=self.paddings[i],
                                    out_padding=self.out_paddings[i])
            self.layer_dict['cond_gated_trans_conv_{}'.format(i)] = conv
            x = conv(x,h)
            print(x.shape)
        
        pre_output_conv = nn.Conv1d(in_channels=x.shape[1], 
                        out_channels=self.pre_output_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.layer_dict['pre_output_conv'] = pre_output_conv
        x = pre_output_conv(x)
        print(x.shape)

        pre_output_bn = nn.BatchNorm1d(self.pre_output_channels)
        self.layer_dict['pre_output_bn'] = pre_output_bn
        x = pre_output_bn(x)
        print(x.shape)

        logit_conv = nn.Conv1d(in_channels=x.shape[1],
                        out_channels=self.num_final_output_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.layer_dict['logit_conv'] = logit_conv
        x = logit_conv(x)
        print(x.shape)
    
    def forward(self, input, speaker):
        num_layers = len(self.kernel_sizes)

        speaker_code = self.speaker_dict(speaker)
        out = input
        for i in range(num_layers):
            out = self.layer_dict['cond_gated_trans_conv_{}'.format(i)](out, speaker_code)
        
        out = self.layer_dict['pre_output_conv'](out)
        out = self.layer_dict['pre_output_bn'](out)
        out = F.relu(out)

        return self.layer_dict['logit_conv'](out)
    
    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()
        
        self.speaker_dict.reset_parameters()


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initializes Vector Quantizer.
        """
        super(VectorQuantizer, self).__init__()
        print('Building VQ layer.')
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.init_embedding()

    def init_embedding(self):
        """
        Initializes the embeddings uniformly.
        """
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

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
        # print(input)
        # print(latents)
        # Change to PyTorch channel first order
        # (batch, time, channel) -> (batch, channel, time)
        quantized_st = quantized_st.permute(0, 2, 1).contiguous()

        # Another quantized output, that is specifically for propagating gradients to update embedding vectors.
        quantized = self.embedding(latents)
        quantized = quantized.permute(0, 2, 1).contiguous()

        return quantized_st, quantized
    
    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.embedding.reset_parameters()
        self.init_embedding()

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

        self.conv_bn = nn.BatchNorm1d(self.out_channels)

        self.gate = nn.Conv1d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=padding)

        self.gate_bn = nn.BatchNorm1d(self.out_channels)
    
    def forward(self, input):
        conv_out = self.conv_bn(self.conv(input))
        gate_out = self.gate_bn(self.gate(input))
        return torch.tanh(conv_out) * torch.sigmoid(gate_out)
    
    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.conv.reset_parameters()
        self.conv_bn.reset_parameters()
        self.gate.reset_parameters()
        self.gate_bn.reset_parameters()

class CondGatedTransposeConv1d(nn.Module):
    """
    Conditional gated 1-dimensional convolution transpose.
    """
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size, stride, dilation=1, padding=0, out_padding=0):
        super(CondGatedTransposeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding=padding
        self.out_padding = out_padding
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.cond = nn.Linear(in_features=self.cond_dim,
                            out_features=self.out_channels)

        self.conv = nn.ConvTranspose1d(in_channels=self.in_channels, 
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=self.padding,
                        output_padding=self.out_padding)

        self.conv_bn = nn.BatchNorm1d(self.out_channels)
        
        self.gate = nn.ConvTranspose1d(in_channels=self.in_channels, 
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=self.padding,
                        output_padding=self.out_padding)

        self.gate_bn = nn.BatchNorm1d(self.out_channels)
        
    def forward(self, input, speaker):
        cond_out = self.cond(speaker).unsqueeze(-1)

        conv_out = self.conv(input)
        conv_out = torch.add(conv_out, cond_out)
        conv_out = self.conv_bn(conv_out)

        gate_out = self.gate(input)
        gate_out = torch.add(gate_out, cond_out)
        gate_out = self.gate_bn(gate_out)

        return torch.tanh(conv_out) * torch.sigmoid(gate_out)

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.cond.reset_parameters()
        self.conv.reset_parameters()
        self.gate.reset_parameters()
        self.conv_bn.reset_parameters()
        self.gate_bn.reset_parameters()
        
class Digital2Analog(nn.Module):
    """
    Converts digital (mu-law encoded) input to continuous input from -1 to 1
    """
    def __init__(self, num_input_quantization_channels):
        super(Digital2Analog, self).__init__()
        self.num_input_quantization_channels = num_input_quantization_channels
        # Dictionary for digital to analog conversion
        self.init_input_embeddings()

    def init_input_embeddings(self):
        self.input_embeddings = nn.Embedding(self.num_input_quantization_channels, 1)

        indices = torch.arange(self.num_input_quantization_channels)
        # Initialize values from -1 to +1 sequentially.
        analog_values = torch.arange(start=-1, end=1, step=1/(self.num_input_quantization_channels/2), requires_grad=False)

        self.input_embeddings.weight.data.put_(indices, analog_values)
        # The dictionary is non-trainable.
        self.input_embeddings.weight.requires_grad = False
    
    def forward(self, input):
        return self.input_embeddings(input).squeeze(-1)

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