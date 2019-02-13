import os
import subprocess
import argparse
import tqdm
import getpass
import time

parser = argparse.ArgumentParser(description='Welcome to the run N at a time script')
parser.add_argument('--num_parallel_jobs', type=int)
parser.add_argument('--total_epochs', type=int)
args = parser.parse_args()


def check_if_experiment_with_name_is_running(experiment_name):
    result = subprocess.run(['squeue --name {}'.format(experiment_name), '-l'], stdout=subprocess.PIPE, shell=True)
    lines = result.stdout.split(b'\n')
    if len(lines) > 2:
        return True
    else:
        return False

student_id = getpass.getuser().encode()[:5]
list_of_scripts = [item for item in
                   subprocess.run(['ls'], stdout=subprocess.PIPE).stdout.split(b'\n') if
                   item.decode("utf-8").endswith(".sh")]

for script in list_of_scripts:
    print('sbatch', script.decode("utf-8"))

epoch_dict = {key.decode("utf-8"): 0 for key in list_of_scripts}
total_jobs_finished = 0

while total_jobs_finished < args.total_epochs * len(list_of_scripts):
    curr_idx = 0
    with tqdm.tqdm(total=len(list_of_scripts)) as pbar_experiment:
        while curr_idx < len(list_of_scripts):
            number_of_jobs = 0
            result = subprocess.run(['squeue', '-l'], stdout=subprocess.PIPE)
            for line in result.stdout.split(b'\n'):
                if student_id in line:
                    number_of_jobs += 1

            if number_of_jobs < args.num_parallel_jobs:
                while check_if_experiment_with_name_is_running(
                        experiment_name=list_of_scripts[curr_idx].decode("utf-8")) or epoch_dict[
                    list_of_scripts[curr_idx].decode("utf-8")] >= args.total_epochs:
                    curr_idx += 1
                    if curr_idx >= len(list_of_scripts):
                        curr_idx = 0

                str_to_run = 'sbatch {}'.format(list_of_scripts[curr_idx].decode("utf-8"))
                total_jobs_finished += 1
                os.system(str_to_run)
                print(str_to_run)
                curr_idx += 1
            else:
                time.sleep(1)
