import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distr
import numpy as np
import time

from experiment_builders.experiment_builder import ExperimentBuilder

EPS = 1e-12

class JointVAEWORLDExperimentBuilder(ExperimentBuilder):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 weight_decay_coefficient, learning_rate, device, continue_from_epoch=-1,
                 print_timings=False):
        super(JointVAEWORLDExperimentBuilder, self).__init__(network_model, experiment_name, num_epochs,
                                                     train_data, val_data, weight_decay_coefficient,
                                                     learning_rate, device, continue_from_epoch, print_timings)

        self.optimizer = optim.Adam(self.parameters(),
                                    amsgrad=False,
                                    lr=learning_rate,
                                    betas=(0.5, 0.999),
                                    weight_decay=weight_decay_coefficient)

    def run_train_iter(self, x, y):
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device)  # send data to device as torch tensors
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        forward_start_time = time.time()
        x_out, latent_dist = self.model.forward(x, y)
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

        # KL objective
        loss_kl = _kl_multiple_discrete_loss(latent_dist, self.device)

        total_loss = loss_kl + loss_recons
        loss_time = time.time() - loss_start_time

        backward_start_time = time.time()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        total_loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        backward_time = time.time() - backward_start_time

        metrics = {}
        metrics['loss'] = total_loss.data.detach().cpu().numpy()
        metrics['loss_recons'] = loss_recons.data.detach().cpu().numpy()
        metrics['loss_kl'] = loss_kl.data.detach().cpu().numpy()
        if self.print_timings:
            metrics['forward_time'] = forward_time
            metrics['backward_time'] = backward_time
            metrics['loss_time'] = loss_time
        return metrics

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode

        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device)  # convert data to pytorch tensors and send to the computation device
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(self.device)
        y = y.to(self.device)

        forward_start_time = time.time()
        x_out, latent_dist = self.model.forward(x, y)
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

        # KL objective
        loss_kl = _kl_multiple_discrete_loss(latent_dist, self.device)

        total_loss = loss_kl + loss_recons
        loss_time = time.time() - loss_start_time

        metrics = {}
        metrics['loss'] = total_loss.data.detach().cpu().numpy()
        metrics['loss_recons'] = loss_recons.data.detach().cpu().numpy()
        metrics['loss_kl'] = loss_kl.data.detach().cpu().numpy()
        if self.print_timings:
            metrics['forward_time'] = forward_time
            metrics['loss_time'] = loss_time
        return metrics

    def convert(self, x, y):
        self.eval()  # sets the system to evaluation mode

        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device)  # convert data to pytorch tensors and send to the computation device
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(self.device)
        y = y.to(self.device)

        x_out, _ = self.model.forward(x, y)  # forward the data through the model
        # The data is scaled to -1 and 1, so we add a tanh activation
        x_out = torch.tanh(x_out)

        return x_out.data


class JointVAERawExperimentBuilder(ExperimentBuilder):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 weight_decay_coefficient, learning_rate, device, continue_from_epoch=-1,
                 print_timings=False):
        super(JointVAERawExperimentBuilder, self).__init__(network_model, experiment_name, num_epochs,
                                                     train_data, val_data, weight_decay_coefficient,
                                                     learning_rate, device, continue_from_epoch, print_timings)

        self.optimizer = optim.Adam(self.parameters(),
                                    amsgrad=False,
                                    lr=learning_rate,
                                    betas=(0.5, 0.999),
                                    weight_decay=weight_decay_coefficient)

    def run_train_iter(self, x, y):
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if type(x) is np.ndarray:
            x = torch.Tensor(x).long().to(device=self.device)  # send data to device as torch tensors
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        forward_start_time = time.time()
        x_out, latent_dist = self.model.forward(x, y)
        forward_time = time.time() - forward_start_time

        loss_start_time = time.time()
        # Reconstruction loss
        loss_recons = F.cross_entropy(x_out, x.squeeze(1))

        # KL objective (mean across the batch!)
        loss_kl = _kl_multiple_discrete_loss(latent_dist, self.device)

        total_loss = loss_kl + loss_recons
        loss_time = time.time() - loss_start_time

        backward_start_time = time.time()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        total_loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        backward_time = time.time() - backward_start_time

        metrics = {}
        metrics['loss'] = total_loss.data.detach().cpu().numpy()
        metrics['loss_recons'] = loss_recons.data.detach().cpu().numpy()
        metrics['loss_kl'] = loss_kl.data.detach().cpu().numpy()
        if self.print_timings:
            metrics['forward_time'] = forward_time
            metrics['backward_time'] = backward_time
            metrics['loss_time'] = loss_time
        return metrics

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode

        if type(x) is np.ndarray:
            x = torch.Tensor(x).long().to(device=self.device)  # convert data to pytorch tensors and send to the computation device
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(self.device)
        y = y.to(self.device)

        forward_start_time = time.time()
        x_out, latent_dist = self.model.forward(x, y)
        forward_time = time.time() - forward_start_time

        loss_start_time = time.time()
        # Reconstruction loss
        loss_recons = F.cross_entropy(x_out, x.squeeze(1))

        # KL objective
        loss_kl = _kl_multiple_discrete_loss(latent_dist, self.device)

        # TODO maybe add hyperparameter for weight to kl term
        total_loss = loss_kl + loss_recons
        loss_time = time.time() - loss_start_time

        metrics = {}
        metrics['loss'] = total_loss.data.detach().cpu().numpy()
        metrics['loss_recons'] = loss_recons.data.detach().cpu().numpy()
        metrics['loss_kl'] = loss_kl.data.detach().cpu().numpy()
        if self.print_timings:
            metrics['forward_time'] = forward_time
            metrics['loss_time'] = loss_time
        return metrics

    def convert(self, x, y):
        self.eval()  # sets the system to evaluation mode

        if type(x) is np.ndarray:
            x = torch.Tensor(x).long().to(device=self.device)  # convert data to pytorch tensors and send to the computation device
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(self.device)
        y = y.to(self.device)

        x_out, _ = self.model.forward(x, y)  # forward the data through the model

        return torch.argmax(x_out.data, dim=1)
    

def _kl_multiple_discrete_loss(alphas, device):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.
    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        torch.Tensor instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    """
    # Calculate kl losses for each discrete latent
    kl_losses = [_kl_discrete_loss(alpha, device) for alpha in alphas]

    # Total loss is sum of kl loss for each discrete latent
    kl_loss = torch.sum(torch.cat(kl_losses))

    return kl_loss

def _kl_discrete_loss(alpha, device):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)]).to(device)
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss