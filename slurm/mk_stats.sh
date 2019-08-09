#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1             
#SBATCH --nodes=1-1            
#SBATCH --cpus-per-task=4            
#SBATCH --mem=40GB              
#SBATCH --time=4:00:00         
#SBATCH --output=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_stats.out
#SBATCH --job-name=mk_stats

python mk_stats.py --overwrite


