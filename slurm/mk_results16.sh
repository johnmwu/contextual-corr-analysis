#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_results16-
OUTPUT_SUFFIX=.out

# maxcorr mincorr maxlinreg minlinreg cca lincka
for method in cca; do
    sbatch --job-name=mk_results16-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_results16-helper.sh $method
done
