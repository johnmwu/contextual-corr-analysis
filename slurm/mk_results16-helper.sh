#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=240GB
# this is only for cca, others 220GB
#SBATCH --time=24:00:00         

method=$1
results="/data/sls/temp/johnmwu/contextual-corr-analysis/results16"
repr_files="repr_files16"
opt_fname="opt16"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}"
