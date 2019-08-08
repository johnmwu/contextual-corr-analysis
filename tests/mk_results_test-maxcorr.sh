#!/bin/bash

results="/data/sls/temp/johnmwu/contextual-corr-analysis/results_test"
repr_files="repr_files_test"
opt_fname="opt_test"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods maxcorr
