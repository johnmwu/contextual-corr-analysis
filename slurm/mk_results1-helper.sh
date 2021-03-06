#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:2           
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=20GB              
#SBATCH --time=4:00:00         

method=$1
results="/data/sls/temp/johnmwu/contextual-corr-analysis/results1"
repr_files="repr_files1"
opt_fname="opt1"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}"
