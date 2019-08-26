#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/belinkov/contextual-corr-analysis/mk_attn_results1-
OUTPUT_SUFFIX=.out

source activate pytorch1.1

# maxcorr mincorr maxlinreg minlinreg cca lincka
#for method in maxcorr mincorr; do
for method in maxcorr; do
    sbatch --job-name=mk_attn_results1-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_attn_results1-helper.sh $method
done
