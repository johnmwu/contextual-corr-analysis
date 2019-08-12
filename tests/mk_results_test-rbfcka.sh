#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:2           
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=10GB              
#SBATCH --time=1:00:00

results="/data/sls/temp/johnmwu/contextual-corr-analysis/results_test"
repr_files="repr_files_test"
opt_fname="opt_test"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods rbfcka --limit 1000
