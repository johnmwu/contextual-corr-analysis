#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_attn_results6
OUTPUT_SUFFIX=.out

# maxcorr mincorr pearsonmaxcorr pearsonmincorr jsmaxcorr jsmincorr
for method in maxcorr mincorr pearsonmaxcorr pearsonmincorr jsmaxcorr jsmincorr; do
    sbatch --job-name=mk_attn_results6-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_attn_results6-helper.sh $method
done
