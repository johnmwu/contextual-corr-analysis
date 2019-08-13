#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_results13-
OUTPUT_SUFFIX=.out

# maxcorr mincorr maxlinreg minlinreg cca lincka
for method in maxcorr mincorr maxlinreg minlinreg cca lincka; do
    sbatch --job-name=mk_results13-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_results13-helper.sh $method
done
