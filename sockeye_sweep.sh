#!/bin/bash

#SBATCH --job-name=Sweep_ASTab_%A_%a
#SBATCH --account=st-puranga-1-gpu
#SBATCH --time=14:30:00
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6                       # CPU cores per MPI process
#SBATCH --mem=32G                               # memory per node! max per GPU is 40G! not 48G
#SBATCH --gpus-per-node=1                       # number of GPUs per node
#SBATCH --constraint=gpu_mem_32                 # Specify the type of GPU required
#SBATCH --output=Sweep_ASTab_%A_%a.out
#SBATCH --error=Sweep_ASTab_%A_%a.err
#SBATCH --mail-user=wuvictoria16@gmail.com
#SBATCH --mail-type=ALL

# Define the range of array indices and step value
#SBATCH --array=1-100

################################################################################
conda init
source ~/.bashrc

module load gcc
#module load python   # do not load! gives error
#module load miniconda3
module load cuda
#module load cudnn   # do not load! gives error
module load nccl

module load http_proxy

conda activate /arc/project/st-puranga-1/users/victoriawu/conda_envs/Torch_AS_Tab

cd /scratch/st-puranga-1/users/victoriawu/workspace/miccai2024/FTC
#cd $SLURM_SUBMIT_DIR

SWEEP_ID="rcl_stroke/as_tab/ru6gn5cd"

python sweep.py