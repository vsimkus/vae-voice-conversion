#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export TEAM_ID=g086

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${TEAM_ID}

export TMPDIR=/disk/scratch/${TEAM_ID}/
export TMP=/disk/scratch/${TEAM_ID}/
export DATASET_DIR=${TMP}/data/

mkdir -p ${DATASET_DIR}

rsync -ua --progress /home/${STUDENT_ID}/data/ ${DATASET_DIR}
unzip -u ${DATASET_DIR}/processed_data.zip -d ${DATASET_DIR}

# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..
python train_evaluate_vqvae.py \
                --use_gpu=True \
                --gpu_id='0' \
                --experiment_name='vqvae_init_exp' \
                --filepath_to_arguments_json_file='experiment_configs/vqvae_architecture.json' \
                --dataset_root_path=${DATASET_DIR}