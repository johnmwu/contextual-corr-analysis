#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/belinkov/contextual-corr-analysis/mk_fine_results3-
OUTPUT_SUFFIX=.out

source activate pytorch1.1

#methods="maxcorr mincorr maxlinreg minlinreg cca lincka"
methods="cca"
for method in $methods; do
    sbatch --job-name=mk_fine_results3-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_fine_results3-helper.sh $method
done
