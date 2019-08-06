#!/bin/bash
#SBATCH --partition=sm
#SBATCH --ntasks=1
#SBATCH --nodes=1-1
#SBATCH --job-name=test_batch
#SBATCH --output=test_batch.out

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_TASKID=$SLURM_ARRAY_TASK_ID"

srun --output=test_batch1.out echo hello & wait

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods mincorr --disable_cuda &

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods maxlinreg --disable_cuda &

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods minlinreg --disable_cuda &

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods cca --disable_cuda &

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods lincka --disable_cuda &

