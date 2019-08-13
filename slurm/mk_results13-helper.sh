#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=240GB              
#SBATCH --time=24:00:00         

method=$1
results="/data/sls/temp/johnmwu/contextual-corr-analysis/results13"
repr_files="repr_files13"
opt_fname="opt13"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}"
