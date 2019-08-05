#!/bin/bash

cd ../
source activate pytorch1.1

input_hdf5_filename="/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/elmo_original/ptb_pos_dev.hdf5"
output_hdf5_filename_pref="/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers"
output_hdf5_filename_suff="ptb_pos_dev.hdf5"
#model_names="xlm-mlm-en-2048 xlm-mlm-ende-1024 xlm-mlm-enfr-1024 xlm-mlm-enro-1024 xlm-clm-enfr-1024 xlm-clm-ende-1024"
model_names="xlm-clm-ende-1024"

for model_name in $model_names ; do 
	echo $model_name
	output_hdf5_filename="$output_hdf5_filename_pref/$model_name/$output_hdf5_filename_suff" ;
	python get_transformer_representations.py $model_name $input_hdf5_filename $output_hdf5_filename ;
done
