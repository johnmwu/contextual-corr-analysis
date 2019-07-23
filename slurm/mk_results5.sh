#!/bin/bash

# Eventually we chose to parallelize the jobs by running each method
# separately. Thus, this script was never run.

results="/data/sls/temp/johnmwu/contextual-corr-analysis/results5"
opt_fname="opt5"

python ../main.py repr_files1 "${results}" --opt_fname "${opt_fname}" --methods maxcorr mincorr maxlinreg minlinreg cca lincka --disable_cuda

