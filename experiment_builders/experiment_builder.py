import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
import sys
from collections import defaultdict

from util.storage_utils import save_statistics

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 weight_decay_coefficient, learning_rate, device, continue_from_epoch=-1, print_timings=False):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param learning_rate: A float indicating the learning rate to use with the adam optimizer.
        :param device: torch.device to use for training the model. (CPU or GPU(s))
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there. If (-2) we'll reload the latest model.
        :param print_timings: A boolean flag whether to print timings for different parts of the training.
        """
        super(ExperimentBuilder, self).__init__()

        self.print_timings = print_timings
        self.experiment_name = experiment_name
        self.model = network_model
        self.model.reset_parameters()
        self.device = device

        if torch.cuda.device_count() > 1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu
          # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optim.Adam(self.parameters(),
                                    amsgrad=False,
                                    lr=learning_rate,
                                    weight_decay=weight_decay_coefficient)
        # Generate the directory names
        self.experiment_root = os.path.abspath('experiments')
        self.experiment_folder = os.path.abspath(os.path.join(self.experiment_root, experiment_name))
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_loss = 0.

        if not os.path.exists(self.experiment_root):  # If experiment root directory does not exist
            os.mkdir(self.experiment_root)  # create the experiment root directory

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        # self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        self.num_epochs = num_epochs
        # Load the latest model
        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']+1
            except:
                print("Model object cannot be found in {}!".format(self.experiment_saved_models))
                sys.exit()

        # Load model from continue_from_epoch
        elif continue_from_epoch != -1:
            self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']+1
        # New model
        else:
            self.starting_epoch = 0
            self.state = dict()

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        raise NotImplementedError

    def run_evaluation_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        raise NotImplementedError

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        print('Loading model {}, at epoch {}.'.format(model_save_name, model_idx))

        file_path = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
        if not (torch.cuda.is_available()):
            map_location='cpu'
        else:
            map_location=None

        state = torch.load(f=file_path, map_location=map_location)

        # This is loads a DataParallel model onto simple model, 
        # Creates a temporary DataParallel model to load the model and then restores to the unwrapped model.
        if not torch.cuda.is_available() and list(state['network'].keys())[0].startswith('model.module'):
            # Temporarily save the unwrapped model
            unwrapped_model = self.model

            # Temporarily wrap the model in DP
            self.model = nn.DataParallel(self.model)
            self.load_state_dict(state['network'])
            
            # Unwrap the model
            self.model = unwrapped_model
        else:
            self.load_state_dict(state['network'])

        return state['best_val_model_idx'], state['best_val_model_loss'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        experiment_metrics = defaultdict(list)
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_metrics = defaultdict(list)

            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                data_fetch_start_time = time.time()
                for idx, (x, y) in enumerate(self.train_data):  # get data batches
                    data_fetch_time = time.time() - data_fetch_start_time
                    metrics = self.run_train_iter(x=x, y=y)  # take a training iter step

                    # Append metrics from the current iteration
                    for key, value in metrics.items():
                        current_epoch_metrics['train_{}'.format(key)].append(value)

                    # Format progress bar description
                    description = (', ').join(['{}: {:.4f}'.format(key, value) for key, value in metrics.items()])
                    if self.print_timings:
                        times = ' data_fetch_time: {:.4f}'.format(data_fetch_time)
                        description = description + times
                    pbar_train.set_description(description)
                    pbar_train.update(1)
                    data_fetch_start_time = time.time()

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                data_fetch_start_time = time.time()
                for x, y in self.val_data:  # get data batches
                    data_fetch_time = time.time() - data_fetch_start_time
                    metrics = self.run_evaluation_iter(x=x, y=y)  # run a validation iter

                    # Append metrics from the current iteration
                    for key, value in metrics.items():
                        current_epoch_metrics['val_{}'.format(key)].append(value)

                    # Format progress bar description
                    description = (', ').join(['{}: {:.4f}'.format(key, value) for key, value in metrics.items()])
                    if self.print_timings:
                        times = ' data_fetch_time: {:.4f}'.format(data_fetch_time)
                        description = description + times
                    pbar_val.set_description(description)
                    pbar_val.update(1)
                    data_fetch_start_time = time.time()

            val_mean_loss = np.mean(current_epoch_metrics['val_loss'])
            if val_mean_loss < self.best_val_model_loss:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_loss = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_metrics.items():
                experiment_metrics[key].append(np.mean(value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            experiment_metrics['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=experiment_metrics, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_metrics.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("\nEpoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            
            if torch.cuda.is_available():
                print('CUDA max allocated memory: {}, max cached memory: {}.'.format(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_cached()))

            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_loss'] = self.best_val_model_loss
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

        return experiment_metrics, test_losses


class GANExperimentBuilder(ExperimentBuilder):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                test_data, weight_decay_coefficient, learning_rate, use_gpu, gpu_id, continue_from_epoch=-1, print_timings=False):
        super(GANExperimentBuilder, self).__init__(network_model, experiment_name, num_epochs,
                train_data, val_data, test_data, weight_decay_coefficient, learning_rate, use_gpu, gpu_id, continue_from_epoch, print_timings)

        self.criterion = nn.BCELoss().to(self.device) # send the loss computation to the GPU

    def run_train_iter(self, x, y):
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device) # send data to device as torch tensors
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        real_correct = torch.Tensor([x.shape[0], 1]).fill_(1.0)
        fake_correct = torch.Tensor([x.shape[0], 1]).fill_(0.0)

        generator_latent = torch.Tensor(np.random.normal(0, 1, (x.shape[0], self.model.generator_arch.latent_dim, self.model.generator_arch.latent_length)))

        forward_start_time = time.time()
        x_hat, real_pred, fake_pred = self.model.forward(x, y, generator_latent)
        forward_time = time.time() - forward_start_time

        loss_start_time = time.time()
        # Binary Cross Entropy - Real
        real_loss = F.bse_loss(real_pred, real_correct)

        # Binary Cross Entropy - Fake
        fake_loss = F.bse_loss(fake_pred, fake_correct)

        total_loss = (real_loss + fake_loss) / 2
        loss_time = time.time() - loss_start_time

        backward_start_time = time.time()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        total_loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        backward_time = time.time() - backward_start_time

        metrics = {}
        metrics['loss'] = total_loss.data.detach().cpu().numpy()
        metrics['real_loss'] = real_loss.data.detach().cpu().numpy()
        metrics['fake_loss'] = fake_loss.data.detach().cpu().numpy()
        if self.print_timings:
            metrics['forward_time'] = forward_time
            metrics['backward_time'] = backward_time
            metrics['loss_time'] = loss_time
        return metrics

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode

        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device) # send data to device as torch tensors
            y = torch.Tensor(y).long().to(device=self.device)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        real_correct = torch.Tensor([x.shape[0], 1]).fill_(1.0)
        fake_correct = torch.Tensor([x.shape[0], 1]).fill_(0.0)

        generator_latent = torch.Tensor(np.random.normal(0, 1, (x.shape[0], self.model.generator_arch.latent_dim, self.model.generator_arch.latent_length)))

        forward_start_time = time.time()
        x_hat, real_pred, fake_pred = self.model.forward(x, y, generator_latent)
        forward_time = time.time() - forward_start_time

        loss_start_time = time.time()
        # Binary Cross Entropy - Real
        real_loss = F.bse_loss(real_pred, real_correct)

        # Binary Cross Entropy - Fake
        fake_loss = F.bse_loss(fake_pred, fake_correct)

        total_loss = (real_loss + fake_loss) / 2
        loss_time = time.time() - loss_start_time
        
        metrics = {}
        metrics['loss'] = total_loss.data.detach().cpu().numpy()
        metrics['real_loss'] = real_loss.data.detach().cpu().numpy()
        metrics['fake_loss'] = fake_loss.data.detach().cpu().numpy()
        if self.print_timings:
            metrics['forward_time'] = forward_time
            metrics['loss_time'] = loss_time
        return metrics

