#!/bin/bash
#SBATCH --job-name=voicecraft                   # Create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/minhld12/output.txt
#SBATCH --error=/lustre/scratch/client/vinai/users/minhld12/error.txt
#SBATCH --partition=research             # Choose partition
#SBATCH --gpus=4                         # GPU count
#SBATCH --nodes=1                        # Node count
#SBATCH --nodelist=sdc2-hpc-dgx-a100-018                       # Node count
#SBATCH --cpus-per-gpu=16                 # CPU cores per GPU
#SBATCH --mem=300G
#SBATCH --mail-type=all          # send email when job fails
#SBATCH --mail-user=v.thivt1@vinai.io

# Your commands here
NVIDIA_DISABLE_REQUIRE=1 srun enroot start \
--mount /lustre/scratch/client/vinai/users/thivt1/code:/lustre/scratch/client/vinai/users/thivt1/code \
--mount /home/thivt1/.cache:/home/thivt1/.cache \
--mount /home/thivt1/.conda:/home/thivt1/.conda \
--mount /home/thivt1/data:/home/thivt1/data \
voicecraft-node-018 \
bash -c "\
    nvidia-smi && \
    echo 'Current location: $(pwd)' && \
    echo 'Initializing environment...' && \
    echo 'ls /root/.miniconda/bin...' && \
    ls -la /root/.miniconda && \
    cat /root/.bashrc && \
    echo 'before adding path' && \
    echo ${PATH} && \
    echo 'after adding path' && \
    export PATH="/root/.miniconda/bin:$PATH"
    echo ${PATH} && \
    echo 'listing conda envs' && \
    conda env list && \
    echo 'Activating Conda environment...' && \
    source activate /conda/voicecraft && \
    echo 'Checking CUDA availability...' && \
    python -c 'import torch; print(torch.cuda.is_available())' && \
    echo 'cd into the code directory...' && \
    cd /lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/z_scripts && \
    echo 'Running training script...' && \
    bash e830M_ft_1500hrs.sh && \
    echo 'All commands executed successfully.' \
"
    # echo 'listing content inside /conda' && \
    # ls /conda/voicecraft && \
    # source /root/.bashrc && \