#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:2           
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=30GB              
#SBATCH --time=12:00:00

method=jsmaxcorr
results="/data/sls/temp/johnmwu/contextual-corr-analysis/attn_results_test6"
repr_files="attn_files_test"
opt_fname="opt_test"

python ../../main_attn.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}"
