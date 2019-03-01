import os.path
import torch

exp_dir = os.path.expanduser('experiments')
for target in sorted(os.listdir(exp_dir)):
    latest_model_path = os.path.join(exp_dir, target, 'saved_models', 'train_model_latest')

    if not os.path.exists(latest_model_path):
        continue

    latest_state = torch.load(f=latest_model_path)
    
    curr_epoch = latest_state['current_epoch_idx']
    best_val_idx = latest_state['best_val_model_idx']
    best_loss = latest_state['best_val_model_loss']

    print('Experiment {} is at epoch: {}, current best val loss: {} at epoch {}'.format(target, curr_epoch, best_loss, best_val_idx))
