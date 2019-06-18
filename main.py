"""
Compute correlations between contextualizer

Usage: 
  main.py METHOD REPRESENTATION_FILES OUTPUT_FILE

Arguments:
  METHOD            Correlation method to use. Choose from one of
                        "min", "max", "linreg", "svcca", "cka"
  REPRESENTATION_FILES   File containing list of locations of representation files 
                        (one per line)
  OUTPUT_FILE       File to write the output correlations in json format

Options:
  -h, --help                             show this help message  

"""

from docopt import docopt
import numpy as np
import json
import os 
import torch
from itertools import product as p
from tqdm import tqdm
from corr_methods import MaxCorr, MinCorr, LinReg, SVCCA, CKA

def main(method, representation_files, output_file):
    print('Initializing method ' + method) 
    if method == 'max':
        method = MaxCorr(representation_files)
    elif method == 'min':
        method = MinCorr(representation_files)
    elif method == 'linreg':
        method = LinReg(representation_files)
    elif method == 'svcca':
        method = SVCCA(representation_files)
    elif method == 'cka':
        method = CKA(representation_files)
    else:
        raise Exception('Unknown method: ' + method)

    print('Loading representations')
    method.load_representations() 

    print('Computing correlations')
    method.compute_correlations()
        
    print('writing correlations to ' + output_file)
    method.write_correlations(output_file)


if __name__ == '__main__':
    args = docopt(__doc__)

    assert args['METHOD'] in {'min', 'max', 'linreg', 'svcca', 'cka'}, 'Unknown METHOD argument: ' + args['METHOD']
    main(args['METHOD'], args['REPRESENTATION_FILES'], args['OUTPUT_FILE']) 

