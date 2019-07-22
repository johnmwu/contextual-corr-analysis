#!/bin/bash

results="/data/sls/temp/johnmwu/contextual-corr-analysis/results8"
repr_files="repr_files8"
opt_fname="opt8"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods rbfcka
