import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_networks import Encoder, Generator
from models.common_layers import LayerNormalizedGatedConv1d, LayerNormalizedGatedTransposeConv1d, LayerNormalizedLReLUConv1d, LayerNormalizedLReLUTransposeConv1d
from models.vq_functions import QuantizeVector

class VQVAE(nn.Module):
    def __init__(self, input_shape, encoder_arch, vq_arch, generator_arch, 
                num_speakers, speaker_dim, use_gated_convolutions=False):
        super(VQVAE, self).__init__()
        self.input_shape = input_shape
        self.encoder_arch = encoder_arch
        self.vq_arch = vq_arch
        self.generator_arch = generator_arch
        self.num_speakers = num_speakers
        self.speaker_dim = speaker_dim
        self.use_gated_convolutions = use_gated_convolutions

        self.speaker_dict = nn.Embedding(num_embeddings=self.num_speakers,
                                        embedding_dim=self.speaker_dim)

        self.build_module()

    def build_module(self):
        print('Building VQVAE.')
        x = torch.zeros((self.input_shape))

        if self.use_gated_convolutions:
            self.encoder = Encoder(input_shape=self.input_shape,
                                kernel_sizes=self.encoder_arch.kernel_sizes,
                                strides=self.encoder_arch.strides,
                                num_output_channels=self.encoder_arch.num_output_channels,
                                paddings=self.encoder_arch.paddings,
                                dilations=self.encoder_arch.dilations,
                                convolution_layer=LayerNormalizedGatedConv1d)
        else:
            self.encoder = Encoder(input_shape=self.input_shape,
                                kernel_sizes=self.encoder_arch.kernel_sizes,
                                strides=self.encoder_arch.strides,
                                num_output_channels=self.encoder_arch.num_output_channels,
                                paddings=self.encoder_arch.paddings,
                                dilations=self.encoder_arch.dilations,
                                convolution_layer=LayerNormalizedLReLUConv1d)

        x = self.encoder(x)

        self.vq = VectorQuantizer(embedding_dim=self.vq_arch.latent_dim,
                                num_embeddings=self.vq_arch.num_latent)

        x_st, x_emb = self.vq(x)
        print('VQ latent shape: {}'.format(x_st.shape))

        # Create speaker embeddings
        y = torch.zeros((self.input_shape[0]), dtype=torch.long)
        y = self.speaker_dict(y)

        self.speaker_dense = nn.Linear(in_features=y.shape[1], out_features=x_st.shape[1]*x_st.shape[2])
        y = self.speaker_dense(y).view(-1, x_st.shape[1], x_st.shape[2])
        print('speaker_out shape: {}'.format(y.shape))

        # Add speaker embeddings
        x_st = x_st + y

        if self.use_gated_convolutions:
            self.generator = Generator(input_shape=x_st.shape,
                                    kernel_sizes=self.generator_arch.kernel_sizes,
                                    strides=self.generator_arch.strides,
                                    dilations=self.generator_arch.dilations,
                                    paddings=self.generator_arch.paddings,
                                    out_paddings=self.generator_arch.out_paddings,
                                    num_output_channels=self.generator_arch.num_output_channels,
                                    convolution_layer=LayerNormalizedGatedTransposeConv1d)
        else:
            self.generator = Generator(input_shape=x_st.shape,
                                    kernel_sizes=self.generator_arch.kernel_sizes,
                                    strides=self.generator_arch.strides,
                                    dilations=self.generator_arch.dilations,
                                    paddings=self.generator_arch.paddings,
                                    out_paddings=self.generator_arch.out_paddings,
                                    num_output_channels=self.generator_arch.num_output_channels,
                                    convolution_layer=LayerNormalizedLReLUTransposeConv1d)

        x_st = self.generator(x_st)

    def forward(self, input, speaker):
        z = self.encoder(input)

        # Straight-through z is passed to decoder, the other is used for computing embedding update
        z_st, z_emb = self.vq(z)

        # Create and add speaker embeddings
        speaker = self.speaker_dict(speaker)
        speaker = self.speaker_dense(speaker).view(-1, z_st.shape[1], z_st.shape[2])
        z_st = z_st + speaker

        x_hat = self.generator(z_st)

        return x_hat, z, z_emb

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        # for item in self.layer_dict.children():
        self.encoder.reset_parameters()
        self.vq.reset_parameters()
        self.generator.reset_parameters()

        self.speaker_dict.reset_parameters()
        self.speaker_dense.reset_parameters()


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
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