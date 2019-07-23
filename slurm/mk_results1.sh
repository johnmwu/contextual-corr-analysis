#!/bin/bash

results="/data/sls/temp/johnmwu/contextual-corr-analysis/results1"
repr_files="repr_files8" # add xlnet
opt_fname="opt1"

python ../main.py "${repr_files}" "${results}" --opt_fname "${opt_fname}" --methods mincorr maxcorr minlinreg maxlinreg cca lincka
