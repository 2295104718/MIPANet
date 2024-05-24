#!/bin/bash

#SBATCH --job-name=sseg
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=47:59:59
#SBATCH --gres=gpu

singularity exec --nv --bind /scratch/$USER --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/PCGNet
python train.py /scratch/$USER/dataset $1 $2 > $1_$2.log 2>&1
"