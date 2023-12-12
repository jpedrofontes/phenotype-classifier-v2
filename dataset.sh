#!/bin/sh

#SBATCH --job-name=pc_v2_dataset
#SBATCH --time=1000:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --nodelist=vision1
#SBATCH --gres=gpu:2
#SBATCH --mem=122000
#SBATCH --cpus-per-task=32
#SBATCH -o /home/mguevaral/jpedro/phenotype-classifier-v2/logs/%x.%j.out 

export TF_GPU_ALLOCATOR=cuda_malloc_async
source /home/mguevaral/jpedro/phenotype-classifier-v2/venv/bin/activate
module load CUDA
module load cuDNN

python /home/mguevaral/jpedro/phenotype-classifier-v2/src/dataset.py /data/mguevaral/crop_bbox/ /home/mguevaral/jpedro/phenotype-classifier-v2/models/val_loss=0.0072.ckpt /data/mguevaral/phenotype-classifier-v2/z_data.csv 
