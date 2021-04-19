#! /bin/csh

#SBATCH -N 1
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_example
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=huishiqiu@gmail.com

module purge
module load 2019
#module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
#module load Anaconda3/2018.12

export PYTHONIOENCODING=utf8
# Activate your environment
conda activate dl2020
# Your job starts in the directory where you call sbatch
cd $HOME/thesis/code/delft_bert/model
#source /home/huishiq/.bashrc
# Run your code
srun python -u dataloader.py

