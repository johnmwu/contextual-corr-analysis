#!/bin/bash
#SBATCH --partition=gpu       
#SBATCH --gres=gpu:2           
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=30GB              
#SBATCH --time=12:00:00         

method=$1
results="/data/sls/temp/johnmwu/contextual-corr-analysis/attn_results6"
repr_files="attn_files6"
opt_fname="attn_opt6"

python ../main_attn.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods "${method}" --ar_mask
