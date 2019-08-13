#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=40GB              
#SBATCH --time=24:00:00

limit=$1
results="/data/sls/temp/johnmwu/contextual-corr-analysis/results12-$limit"
repr_files="repr_files12"
opt_fname="opt12"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods rbfcka --limit=$limit
