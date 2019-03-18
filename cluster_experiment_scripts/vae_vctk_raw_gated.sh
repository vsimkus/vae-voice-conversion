#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Interactive
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-02:00:00

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

mkdir -p ${TMP}/data_small_samples/

# Sync VCTKRaw dataset
rsync -ua --progress /home/${STUDENT_ID}/data/vctk_small_samples.zip ${TMP}/data_small_samples/
unzip -uo ${TMP}/data_small_samples/vctk_small_samples.zip -d ${TMP}/data_small_samples

export DATASET_DIR=${TMP}/data_small_samples/

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..

config_file='vae_vctk_raw_gated.json'
echo "Starting train_vae.py on ${config_file}"
# export PYTHONUNBUFFERED=TRUE # This allows to dump the log messages into stdout immediately
python train_vae.py \
                --use_gpu=True \
                --gpu_id='0,1' \
                --filepath_to_arguments_json_file="experiment_configs/${config_file}" \
                --dataset_root_path=${DATASET_DIR} \
                --print_timings=True
