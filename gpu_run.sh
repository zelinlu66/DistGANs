#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_1g.5gb
#SBATCH -A cmsc714-class
#SBATCH -J DisGAN
#SBATCH -t 01:00:00

# load necessary libraries
#module unload cuda/gcc/11.3.0/zen2/12.3.0
#module unload gcc/11.3.0
#module unload gcc
#module load gcc/9.4.0
#module load opencv/gcc/9.4.0/openmpi/4.1.1/zen2/4.5.2

module load cuda
module load pytorch

# modules doesn't add lib64 path to LD_LIBRARY_PATH for some reason
# do it manually for cudart and opencv
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/cuda-11.6.2-eonihhhvlh4s2d6riyb7al2qivzn477u/lib64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/opencv-4.5.2-xxyodykxk3vuw64tlvm6sujgaxnctgep/lib64"

# activate virture environment
source repo/bin/activate

python main_GANs.py -e 2
