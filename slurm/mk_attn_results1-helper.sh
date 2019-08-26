#!/bin/bash
#SBATCH --partition=2080 
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=50GB              
#SBATCH --time=24:00:00         

method=$1
results="/data/sls/temp/belinkov/contextual-corr-analysis/attn_results1"
repr_files="attn_files1"
opt_fname="attn_opt1"

python ../main_attn.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}" --limit 10 
