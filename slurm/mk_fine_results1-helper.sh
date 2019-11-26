#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=240GB              
#SBATCH --time=2-23:59:00         

method=$1
results="/data/sls/temp/belinkov/contextual-corr-analysis/repr_fine_results1"
repr_files="repr_fine_files1"
opt_fname="repr_fine_opt1"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}"  
