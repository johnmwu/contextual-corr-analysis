#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_attn_results6
OUTPUT_SUFFIX=.out

# maxcorr mincorr pearsonmaxcorr pearsonmincorr jsmaxcorr jsmincorr attn_lincka attn_cca
for method in attn_lincka attn_cca; do
    sbatch --job-name=mk_attn_results6-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_attn_results6-helper.sh $method
done
