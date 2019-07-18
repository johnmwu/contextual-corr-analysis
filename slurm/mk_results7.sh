#!/bin/bash
#SBATCH --partition=630
#SBATCH --ntasks=2
#SBATCH --nodes=1-2
#SBATCH --job-name=contex7
#SBATCH --mem=98GB
#SBATCH --output=mk_results7.out

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_TASKID=$SLURM_ARRAY_TASK_ID"

results7="/data/sls/temp/johnmwu/contextual-corr-analysis/results7"
opt_fname="opt5"

echo "$HDF5_USE_FILE_LOCKING"

srun --output=temp1.out --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods maxcorr --disable_cuda & \
srun --output=temp2.out --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods mincorr --disable_cuda & \
wait

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods maxlinreg --disable_cuda &

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods minlinreg --disable_cuda &

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods cca --disable_cuda &

# srun --cpus-per-task=4 --mem=98GB python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods lincka --disable_cuda &

