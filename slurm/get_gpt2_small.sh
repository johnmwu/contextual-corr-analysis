#!/bin/bash

cd ../
source activate pytorch1.1

input_hdf5_filename="/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/elmo_original/ptb_pos_dev.hdf5"
output_hdf5_filename="/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/gpt2_small/ptb_pos_dev.hdf5"
model_name="gpt2"


python get_transformer_representations.py $model_name $input_hdf5_filename $output_hdf5_filename
