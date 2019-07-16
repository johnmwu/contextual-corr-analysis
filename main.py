import argparse
from corr_methods import (load_representations, MaxCorr, MinCorr,
                          MaxLinReg, MinLinReg, CCA, LinCKA)

def get_options(opt_fname):
    if opt_fname == None:
        layerspec_l = [-1] * len(representation_fname_l)
        first_half_only_l = [False] * len(representation_fname_l)
        second_half_only_l = [False] * len(representation_fname_l)
    else:
        with open(opt_fname, 'r') as f:
            opt_l = [line.strip().split(',') for line in f]
            l, f, s = zip(*opt_l)

            layerspec_l = []
            for ls in l:
                if ls == "all":
                    layerspec_l.append(ls)
                else:
                    layerspec_l.append(int(ls))

            first_half_only_l = []
            for fho in f:
                if fho == 't':
                    first_half_only_l.append(True)
                else:
                    first_half_only_l.append(False)

            second_half_only_l = []
            for sho in s:
                if sho == 't':
                    second_half_only_l.append(True)
                else:
                    second_half_only_l.append(False)

    return layerspec_l, first_half_only_l, second_half_only_l

def get_method_l(methods, num_neurons_d, representations_d):
    if 'all' in methods:
        method_l = [
            MaxCorr(num_neurons_d, representations_d),
            MinCorr(num_neurons_d, representations_d),
            MaxLinReg(num_neurons_d, representations_d),
            MinLinReg(num_neurons_d, representations_d),
            CCA(num_neurons_d, representations_d),
            LinCKA(num_neurons_d, representations_d),
            ]
    else:
        method_l = []
        for method in methods:
            if method == 'maxcorr':
                method_l.append(MaxCorr(num_neurons_d, representations_d))
            elif method == 'mincorr':
                method_l.append(MinCorr(num_neurons_d, representations_d))
            elif method == 'maxlinreg':
                method_l.append(MaxLinReg(num_neurons_d, representations_d))
            elif method == 'minlinreg':
                method_l.append(MinLinReg(num_neurons_d, representations_d))
            elif method == 'cca':
                method_l.append(CCA(num_neurons_d, representations_d))
            elif method == 'lincka':
                method_l.append(LinCKA(num_neurons_d, representations_d))

    return method_l

def main(methods, representation_files, output_file, opt_fname=None,
         limit=None, disable_cuda=False):

    # Set `representation_fname_l`, and options
    with open(representation_files) as f:
        representation_fname_l = [line.strip() for line in f]

    layerspec_l,first_half_only_l,second_half_only_l = get_options(opt_fname)
    
    # Load
    print("Loading representations")
    num_neurons_d, representations_d = load_representations(representation_fname_l, limit=limit,
                                                            layerspec_l=layerspec_l, first_half_only_l=first_half_only_l,
                                                            second_half_only_l=second_half_only_l, disable_cuda=disable_cuda)
    
    # Set `method_l`, list of Method objects
    print('\nInitializing methods ' + str(methods))
    method_l = get_method_l(methods, num_neurons_d, representations_d)

    # Run all methods in method_l
    print('\nComputing correlations')
    for method in method_l:
        print('For method: ', str(method))
        method.compute_correlations()

    print('\nWriting correlations')
    for method in method_l:
        print('For method: ', str(method))
        out_fname = (output_file + '_' + str(method) if len(method_l) > 1
                     else output_file)
        method.write_correlations(out_fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("representation_files")
    parser.add_argument("output_file")
    parser.add_argument("--opt_fname", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_cuda", action="store_true")

    args = parser.parse_args()
    main(args.methods, args.representation_files, args.output_file,
         opt_fname=args.opt_fname, limit=args.limit,
         disable_cuda=args.disable_cuda) 


