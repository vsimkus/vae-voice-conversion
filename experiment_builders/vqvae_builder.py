import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distr
import numpy as np
import time

from experiment_builders.experiment_builder import ExperimentBuilder

class VQVAEWORLDExperimentBuilder(ExperimentBuilder):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                weight_decay_coefficient, learning_rate, commit_coefficient, device, continue_from_epoch=-1, print_timings=False):
        super(VQVAEWORLDExperimentBuilder, self).__init__(network_model, experiment_name, num_epochs,
                train_data, val_data, weight_decay_coefficient, learning_rate, device, continue_from_epoch, print_timings)

        self.commit_coefficient = commit_coefficient

    def run_train_iter(self, x, y):
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device) # send data to device as torch tensors
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        forward_start_time = time.time()
        x_out, z_encoder, z_emb = self.model.forward(x, y)
        # The data is scaled to -1 and 1, so we add a tanh activation
        x_out = torch.tanh(x_out)
        forward_time = time.time() - forward_start_time

        loss_start_time = time.time()
        # Reconstruction loss
        # Flatten x
        x = x.view(x.shape[0], -1)
        x_out = x_out.view(x_out.shape[0], -1)
        # Construct Gaussian with mean as input x
        # Use Normal, not MultivariateNormal because it's *much* faster and our covariance is an identity anyway
        gaussian = distr.Normal(loc=x, scale=1)
        # Compute log likelihood that our model generates x_out from a Gaussian with mean x,
        # and sum over timesteps
        log_prob = gaussian.log_prob(x_out).sum(1)
        # Mean across batches - this is our reconstruction objective
        loss_recons = -log_prob.mean()

        # Vector quantization objective
        loss_vq = F.mse_loss(z_emb, z_encoder.detach())

        # Commitment objective
        loss_commit = F.mse_loss(z_encoder, z_emb.detach())

        total_loss = loss_recons + loss_vq + self.commit_coefficient * loss_commit
        loss_time = time.time() - loss_start_time

        backward_start_time = time.time()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        total_loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        backward_time = time.time() - backward_start_time

        metrics = {}
        metrics['loss'] = total_loss.data.detach().cpu().numpy()
        metrics['loss_recons'] = loss_recons.data.detach().cpu().numpy()
        metrics['loss_vq'] = loss_vq.data.detach().cpu().numpy()
        metrics['loss_commit'] = loss_commit.data.detach().cpu().numpy()
        if self.print_timings:
            metrics['forward_time'] = forward_time
            metrics['backward_time'] = backward_time
            metrics['loss_time'] = loss_time
        return metrics

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode

        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device) # convert data to pytorch tensors and send to the computation device
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(self.device)
        y = y.to(self.device)

        forward_start_time = time.time()
        x_out, z_encoder, z_emb = self.model.forward(x, y)  # forward the data in the model
        # The data is scaled to -1 and 1, so we add a tanh activation
        x_out = torch.tanh(x_out)
        forward_time = time.time() - forward_start_time

        loss_start_time = time.time()
        # Reconstruction loss
        # Flatten x
        x = x.view(x.shape[0], -1)
        x_out = x_out.view(x_out.shape[0], -1)
        # Construct Gaussian with mean as input x
        # Use Normal, not MultivariateNormal because it's *much* faster and our covariance is an identity anyway
        gaussian = distr.Normal(loc=x, scale=1)
        # Compute log likelihood that our model generates x_out from a Gaussian with mean x,
        # and sum over timesteps
        log_prob = gaussian.log_prob(x_out).sum(1)
        # Mean across batches - this is our reconstruction objective
        loss_recons = -log_prob.mean()

        # Vector quantization objective
        loss_vq = F.mse_loss(z_emb, z_encoder.detach())

        # Commitment objective
        loss_commit = F.mse_loss(z_encoder, z_emb.detach())

        total_loss = loss_recons + loss_vq + self.commit_coefficient * loss_commit
        loss_time = time.time() - loss_start_time

        metrics = {}
        metrics['loss'] = total_loss.data.detach().cpu().numpy()
        metrics['loss_recons'] = loss_recons.data.detach().cpu().numpy()
        metrics['loss_vq'] = loss_vq.data.detach().cpu().numpy()
        metrics['loss_commit'] = loss_commit.data.detach().cpu().numpy()
        if self.print_timings:
            metrics['forward_time'] = forward_time
            metrics['loss_time'] = loss_time
        return metrics

    def convert(self, x, y):
        self.eval()  # sets the system to evaluation mode

        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device) # convert data to pytorch tensors and send to the computation device
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(self.device)
        y = y.to(self.device)

        x_out, _, _ = self.model.forward(x, y)  # forward the data in the model
        # The data is scaled to -1 and 1, so we add a tanh activation
        x_out = torch.tanh(x_out)

        return x_out.data