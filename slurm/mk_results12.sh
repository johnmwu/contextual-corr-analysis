#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_results12-
OUTPUT_SUFFIX=.out

# 5000 10000 15000 20000
for limit in 5000 10000 15000; do
    sbatch --job-name=mk_results12-$limit --output="${OUTPUT_PREFIX}${limit}${OUTPUT_SUFFIX}" mk_results12-helper.sh $limit
done
