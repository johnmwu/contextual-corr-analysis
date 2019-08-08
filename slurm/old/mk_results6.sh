#!/bin/bash

results6="/data/sls/temp/johnmwu/contextual-corr-analysis/results6"
opt_fname="opt6"

python ../main.py repr_files6 "${results6}" --opt_fname "${opt_fname}" --methods maxcorr 

