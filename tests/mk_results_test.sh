#!/bin/bash

OUTPUT_PREFIX=/data/sls/temp/johnmwu/contextual-corr-analysis/mk_results_test_
OUTPUT_SUFFIX=.out

# for method in maxcorr mincorr maxlinreg minlinreg cca lincka; do
#     sbatch --job-name=mk_results_test-$method --output="${OUTPUT_PREFIX}${method}${OUTPUT_SUFFIX}" mk_results_test-helper.sh $method
# done

sbatch --job-name=mk_results_test-rbfcka --output="${OUTPUT_PREFIX}rbfcka${OUTPUT_SUFFIX}" mk_results_test-rbfcka.sh
