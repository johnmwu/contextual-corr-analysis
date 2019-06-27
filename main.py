"""
Compute correlations between contextualizer

Usage: 
  main.py METHOD REPRESENTATION_FILES OUTPUT_FILE

Arguments:
  METHOD            Correlation method to use. Choose from one of
                        "min", "max", "linreg", "svcca", "cka", "all"
  REPRESENTATION_FILES   File containing list of locations of representation files 
                        (one per line)
  OUTPUT_FILE       File to write the output correlations

Options:
  -h, --help                             show this help message  

"""

from docopt import docopt

def main(method, representation_files, output_file):
    from corr_methods import load_representations, MaxCorr, MinCorr, LinReg, SVCCA, CKA

    with open(representation_files) as f:
        representation_fname_l = [line.strip() for line in f]

    print("Loading representations")
    num_neurons_d, representations_d = load_representations(representation_fname_l)
    
    print('\nInitializing method ' + method) 
    if method == 'all':
        methods = [
            MaxCorr(num_neurons_d, representations_d),
            MinCorr(num_neurons_d, representations_d),
            LinReg(num_neurons_d, representations_d),
            SVCCA(num_neurons_d, representations_d),
            CKA(num_neurons_d, representations_d),
                   ]
    elif method == 'max':
        methods = [MaxCorr(num_neurons_d, representations_d)]
    elif method == 'min':
        methods = [MinCorr(num_neurons_d, representations_d)]
    elif method == 'linreg':
        methods = [LinReg(num_neurons_d, representations_d)]
    elif method == 'svcca':
        methods = [SVCCA(num_neurons_d, representations_d)]
    elif method == 'cka':
        methods = [CKA(num_neurons_d, representations_d)]
    else:
        raise Exception('Unknown method: ' + method)

    print('\nComputing correlations')
    for method in methods:
        print('For method: ', str(method))
        method.compute_correlations()

    print('\nWriting correlations')
    for method in methods:
        print('For method: ', str(method))
        out_fname = (output_file + '_' + str(method) if len(methods) > 1
                     else output_file)
        method.write_correlations(out_fname)

if __name__ == '__main__':
    args = docopt(__doc__)

    assert args['METHOD'] in {'min', 'max', 'linreg', 'svcca', 'cka', 'all'}, 'Unknown METHOD argument: ' + args['METHOD']
    main(args['METHOD'], args['REPRESENTATION_FILES'], args['OUTPUT_FILE']) 


