#!/bin/bash

results5="/data/sls/temp/johnmwu/contextual-corr-analysis/results5"
opt_fname="opt5"

python ../main.py repr_files1 "${results5}" --opt_fname "${opt_fname}" --methods maxcorr mincorr maxlinreg minlinreg cca lincka --disable_cuda

