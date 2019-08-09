#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:2           
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=140GB              
#SBATCH --time=12:00:00         

method=$1
results="/data/sls/temp/johnmwu/contextual-corr-analysis/results8"
repr_files="repr_files8"
opt_fname="opt8"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}"
