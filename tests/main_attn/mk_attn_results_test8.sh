#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_attn_results_test8
OUTPUT_SUFFIX=.out

sbatch --job-name=mk_attn_results_test8 --output="${OUTPUT_PREFIX}${OUTPUT_SUFFIX}" mk_attn_results_test8-helper.sh
