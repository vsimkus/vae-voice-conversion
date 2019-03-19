import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_networks import Encoder, Generator
from models.common_layers import LayerNormalizedGatedConv1d, LayerNormalizedGatedTransposeConv1d, LayerNormalizedLReLUConv1d, LayerNormalizedLReLUTransposeConv1d

EPS = 1e-12

class JointVAE(nn.Module):
    """
    Implementation of 'Learning Disentangled Joint Continuous and Discrete Representations' by E. Dupont
    https://arxiv.org/abs/1804.00104

    Code based on https://github.com/Schlumberger/joint-vae/blob/master/jointvae/models.py
    """
    def __init__(self, input_shape, encoder_arch, generator_arch, latent_dim, num_latents, temperature,
                num_speakers, speaker_dim, use_gated_convolutions=False):
        super(JointVAE, self).__init__()
        self.input_shape = input_shape
        self.encoder_arch = encoder_arch
        self.latent_dim = latent_dim
        self.latent_param_dim = latent_dim*num_latents
        self.num_latents = num_latents
        self.temperature = temperature
        self.generator_arch = generator_arch
        self.num_speakers = num_speakers
        self.speaker_dim = speaker_dim
        self.use_gated_convolutions = use_gated_convolutions

        self.speaker_dict = nn.Embedding(num_embeddings=self.num_speakers,
                                        embedding_dim=self.speaker_dim)

        self.build_module()

    def build_module(self):
        print('Building JointVAE.')
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

        # Map encoded features to a hidden latent dimension that will be used to encode parameters of the latent distribution
        self.encoded_to_latent = nn.Linear(in_features=z_e.shape[-1], out_features=self.latent_dim) #-> hidden_dim

        z = self.encoded_to_latent(z_e)
        print('Latent dim: {}'.format(z.shape))

        # Encode parameters of latent distribution
        alphas = []
        for _ in range(self.latent_dim):
            alphas.append(nn.Linear(self.latent_dim, self.num_latents))
        self.alphas = nn.ModuleList(alphas)

        latent_dist = self.encode_latent_parameters(z)
        print('Latent distribution: {}'.format(len(latent_dist)))
        latent_sample = self.reparameterize(latent_dist)
        print('Latent sample: {}'.format(latent_sample))

        # Map latent samples to features for generative model
        self.latent_to_generator = nn.Sequential(
            nn.Linear(in_features=self.latent_param_dim, out_features=self.latent_dim),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(in_features=self.latent_dim, out_features=z_e.shape[-1]),
            nn.LeakyReLU(negative_slope=0.02)
        )

        z = self.latent_to_generator(latent_sample)
        print('Latent out: {}'.format(z.shape))

        # Speaker conditioning
        y = torch.zeros((self.input_shape[0]), dtype=torch.long)
        y = self.speaker_dict(y)

        self.speaker_dense = nn.Linear(in_features=y.shape[1], out_features=z.shape[-1])
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

        # Map encoded features to a hidden latent dimension that will be used to encode parameters of the latent distribution
        z = self.encoded_to_latent(z_e)

        # Sample latent
        latent_dist = self.encode_latent_parameters(z)
        latent_sample = self.reparameterize(latent_dist)

        # Map latent samples to features for generative model
        z = self.latent_to_generator(latent_sample)

        y = self.speaker_dense(self.speaker_dict(speaker))
        z = z + y

        # reshape back to 3D tensor
        z = z.view(-1, self.z_e_shape[1], self.z_e_shape[2])

        x_hat = self.generator(z)

        return x_hat, latent_dist

    def encode_latent_parameters(self, latent):
        # Output parameters of latent distribution from hidden representation
        latent_dist = []

        for alpha in self.alphas:
            latent_dist.append(F.softmax(alpha(latent), dim=1))
        
        return latent_dist
    
    def reparameterize(self, latent_dist):
        latent_sample = []

        for alpha in latent_dist:
            disc_sample = self.sample_gumbel_softmax(alpha)
            latent_sample.append(disc_sample)

        # Concatenate continuous and discrete samples into one large sample
        return torch.cat(latent_sample, dim=1)

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization trick.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand_like(alpha.size())
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)

        return one_hot_samples

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.encoder.reset_parameters()
        self.generator.reset_parameters()
        self.encoded_to_latent.reset_parameters()
        self.speaker_dense.reset_parameters()
        for layer in self.alphas:
            layer.reset_parameters()
        
        for layer in self.latent_to_generator:
            if not isinstance(layer, nn.LeakyReLU):
                layer.reset_parameters()

        self.speaker_dict.reset_parameters()