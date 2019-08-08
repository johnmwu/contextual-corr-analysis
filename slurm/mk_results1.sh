#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_results1-1-
OUTPUT_SUFFIX=.out

for method in maxcorr mincorr maxlinreg minlinreg cca lincka; do
    sbatch --job-name=mk_results1-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_results1-helper.sh $method
done
