#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=128GB              
#SBATCH --time=2-23:59:00         

method=$1
results="/data/sls/temp/belinkov/contextual-corr-analysis/repr_fine_results2"
repr_files="repr_fine_files2-small"
opt_fname="repr_fine_opt2-small"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}" 
