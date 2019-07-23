#!/bin/bash

results="/data/sls/temp/johnmwu/contextual-corr-analysis/results9"
repr_files="repr_files1"
opt_fname="opt1"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods rbfcka
