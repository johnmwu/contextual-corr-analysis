import argparse
from corr_methods import (load_representations, MaxCorr, MinCorr,
                          MaxLinReg, MinLinReg, SVCCA, CKA)

def main(method, representation_files, output_file, limit=None,
         disable_cuda=False):
    with open(representation_files) as f:
        representation_fname_l = [line.strip() for line in f]

    print("Loading representations")
    num_neurons_d, representations_d = load_representations(representation_fname_l, limit=limit, disable_cuda=disable_cuda)
    
    print('\nInitializing method ' + method) 
    if method == 'all':
        methods = [
            MaxCorr(num_neurons_d, representations_d),
            MinCorr(num_neurons_d, representations_d),
            MaxLinReg(num_neurons_d, representations_d),
            MinLinReg(num_neurons_d, representations_d),
            SVCCA(num_neurons_d, representations_d),
            CKA(num_neurons_d, representations_d),
            ]
    elif method == 'maxcorr':
        methods = [MaxCorr(num_neurons_d, representations_d)]
    elif method == 'mincorr':
        methods = [MinCorr(num_neurons_d, representations_d)]
    elif method == 'maxlinreg':
        methods = [MaxLinReg(num_neurons_d, representations_d)]
    elif method == 'minlinreg':
        methods = [MinLinReg(num_neurons_d, representations_d)]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("method",
                        choices={'mincorr', 'maxcorr', 'maxlinreg',
                                           'minlinreg', 'svcca', 'cka',
                                           'all'})
    parser.add_argument("representation_files")
    parser.add_argument("output_file")
    parser.add_argument("--limit", dest="limit", type=int, default=None)
    parser.add_argument("--disable_cuda", action="store_true")

    args = parser.parse_args()
    main(args.method, args.representation_files, args.output_file,
         limit=args.limit, disable_cuda=args.disable_cuda) 


