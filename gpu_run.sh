#!/bin/bash
#SBATCH -N 1                            # Request 1 node
#SBATCH -p gpu                          # Use the GPU partition
#SBATCH --gres=gpu:a100:4       # Request 2 GPUs
#SBATCH --mem=128G
#SBATCH -A cmsc714-class               # Your account
#SBATCH -J DistGAN                     # Job name
#SBATCH -t 01:00:00                    # Time limit
#SBATCH --output=logs/distgan_job-%j_node-%N_rank-%t.out


module load cuda
module load pytorch

# Fix LD_LIBRARY_PATH manually for cudart and opencv
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/cuda-11.6.2-eonihhhvlh4s2d6riyb7al2qivzn477u/lib64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/opencv-4.5.2-xxyodykxk3vuw64tlvm6sujgaxnctgep/lib64"

# Activate Python virtual environment
# source $HOME/miniconda/bin/activate
# conda activate distgan
export PATH=$HOME/miniconda/envs/distgan/bin:$PATH

# Recommended environment variables for NCCL and debugging
export OMP_NUM_THREADS=4
export NCCL_DEBUG=warn
export PYTHONFAULTHANDLER=1

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


$CONDA_PREFIX/bin/python -m torch.distributed.run --nproc_per_node=4 main_GANs.py -e 20