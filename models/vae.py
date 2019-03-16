import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_networks import Encoder, Generator
from models.common_layers import LayerNormalizedGatedConv1d, LayerNormalizedGatedTransposeConv1d, LayerNormalizedLReLUConv1d, LayerNormalizedLReLUTransposeConv1d

class VAE(nn.Module):
    def __init__(self, input_shape, encoder_arch, generator_arch, latent_dim,
                num_speakers, speaker_dim, use_gated_convolutions=False):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.encoder_arch = encoder_arch
        self.latent_dim = latent_dim
        self.generator_arch = generator_arch
        self.num_speakers = num_speakers
        self.speaker_dim = speaker_dim
        self.use_gated_convolutions = use_gated_convolutions

        self.speaker_dict = nn.Embedding(num_embeddings=self.num_speakers,
                                        embedding_dim=self.speaker_dim)

        self.build_module()

    def build_module(self):
        print('Building VAE.')
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
        z_e = self.encoder(x)
        self.z_e_shape = z_e.shape

        # Flatten z_e
        z_e = z_e.view(z_e.shape[0], -1)

        # Mean of the posterior
        self.mean_dense = nn.Linear(in_features=z_e.shape[1], out_features=self.latent_dim)
        mean = self.mean_dense(z_e)
        print('mean shape: {}'.format(mean.shape))

        # Log-variance of the posterior
        self.logvar_dense = nn.Linear(in_features=z_e.shape[1], out_features=self.latent_dim)
        logvar = self.logvar_dense(z_e)
        print('logvar shape: {}'.format(logvar.shape))

        # Sample latent from the posterior
        z = self.sample_latent(mean, logvar)
        print('latent shape: {}'.format(z.shape))

        self.latent_dense = nn.Linear(in_features=z.shape[1], out_features=self.z_e_shape[1]*self.z_e_shape[2])
        z = self.latent_dense(z)
        print('latent_out shape: {}'.format(z.shape))

        # Speaker conditioning
        y = torch.zeros((self.input_shape[0]), dtype=torch.long)
        y = self.speaker_dict(y)

        self.speaker_dense = nn.Linear(in_features=y.shape[1], out_features=self.z_e_shape[1]*self.z_e_shape[2])
        y = self.speaker_dense(y)
        print('speaker_out shape: {}'.format(y.shape))

        # Add speaker embedding to the latent
        z = z+y

        # reshape back to 3D tensor
        z = z.view(self.z_e_shape)
        print('latent_out reshaped: {}'.format(z.shape))

        if self.use_gated_convolutions:
            self.generator = Generator(input_shape=z.shape,
                                    kernel_sizes=self.generator_arch.kernel_sizes,
                                    strides=self.generator_arch.strides,
                                    dilations=self.generator_arch.dilations,
                                    paddings=self.generator_arch.paddings,
                                    out_paddings=self.generator_arch.out_paddings,
                                    num_output_channels=self.generator_arch.num_output_channels,
                                    convolution_layer=LayerNormalizedGatedTransposeConv1d)
        else:
            self.generator = Generator(input_shape=z.shape,
                                    kernel_sizes=self.generator_arch.kernel_sizes,
                                    strides=self.generator_arch.strides,
                                    dilations=self.generator_arch.dilations,
                                    paddings=self.generator_arch.paddings,
                                    out_paddings=self.generator_arch.out_paddings,
                                    num_output_channels=self.generator_arch.num_output_channels,
                                    convolution_layer=LayerNormalizedLReLUTransposeConv1d)

        x_hat = self.generator(z)


    def forward(self, input, speaker):
        z_e = self.encoder(input)

        # Flatten z_e
        z_e = z_e.view(z_e.shape[0], -1)

        # Get variational parameters
        mean = self.mean_dense(z_e)
        logvar = self.logvar_dense(z_e)

        # Sample from posterior if during training, otherwise just pass the variational mean
        if self.training:
            z = self.sample_latent(mean, logvar)
        else:
            z = mean

        # Form latent output, add speaker embeddings
        z = self.latent_dense(z)
        y = self.speaker_dense(self.speaker_dict(speaker))
        z = z + y

        # reshape back to 3D tensor
        z = z.view(-1, self.z_e_shape[1], self.z_e_shape[2])

        x_hat = self.generator(z)

        return x_hat, mean, logvar

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.encoder.reset_parameters()
        self.generator.reset_parameters()
        self.mean_dense.reset_parameters()
        self.logvar_dense.reset_parameters()
        self.latent_dense.reset_parameters()
        self.speaker_dense.reset_parameters()

        self.speaker_dict.reset_parameters()

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std