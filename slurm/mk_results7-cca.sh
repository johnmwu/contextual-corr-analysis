#!/bin/bash

results7="/data/sls/temp/johnmwu/contextual-corr-analysis/results7"
opt_fname="opt5"

python ../main.py repr_files1 "${results7}" --opt_fname "${opt_fname}" --methods cca --disable_cuda
