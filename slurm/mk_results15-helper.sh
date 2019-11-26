#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=100GB              
#SBATCH --time=24:00:00         

method=$1
results="/data/sls/temp/johnmwu/contextual-corr-analysis/results15"
repr_files="repr_files15"
opt_fname="opt15"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}"
