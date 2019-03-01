#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:6
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00
#SBATCH --exclude=ladonia[20,24]

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
export TMP=/disk/scratch/${TEAM_ID}

mkdir -p ${TMP}/data/

rsync -ua --progress /home/${STUDENT_ID}/data/vctk.zip ${TMP}/data/
unzip -u ${TMP}/data/vctk.zip -d ${TMP}/data

export DATASET_DIR=${TMP}/data/

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..
echo "Starting train_vqvae.py"
# export PYTHONUNBUFFERED=TRUE # This allows to dump the log messages into stdout immediately
python train_vqvae.py \
                --tuning_mode=True \
                --use_gpu=True \
                --gpu_id='0,1,2,3,4,5' \
                --experiment_name='vqvae_init_exp_small' \
                --filepath_to_arguments_json_file='experiment_configs/vqvae_architecture.json' \
                --dataset_root_path=${DATASET_DIR} 
