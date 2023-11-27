#!/bin/sh

#SBATCH --job-name=pc_v2_ae
#SBATCH --time=1000:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --nodelist=vision1
#SBATCH --gres=gpu:1
#SBATCH --mem=122000
#SBATCH --cpus-per-task=32
#SBATCH -o /home/mguevaral/jpedro/phenotype-classifier-v2/logs/%x.%j.out 

export TF_GPU_ALLOCATOR=cuda_malloc_async
source /home/mguevaral/jpedro/phenotype-classifier-v2/venv/bin/activate
module load CUDA
module load cuDNN

python /home/mguevaral/jpedro/phenotype-classifier-v2/src/train.py -d /data/mguevaral/crop_bbox_sample/ -o models/ -l logs/ 
