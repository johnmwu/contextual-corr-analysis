#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_attn_results5
OUTPUT_SUFFIX=.out

# maxcorr mincorr pearsonmaxcorr pearsonmincorr jsmaxcorr jsmincorr
for method in jsmaxcorr jsmincorr; do
    sbatch --job-name=mk_attn_results5-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_attn_results5-helper.sh $method
done
