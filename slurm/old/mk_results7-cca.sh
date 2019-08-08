#!/bin/bash

results="/data/sls/temp/johnmwu/contextual-corr-analysis/results7"
repr_files="repr_files3"
opt_fname="opt5"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods cca --disable_cuda
