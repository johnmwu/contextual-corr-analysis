import torch
import argparse
from attention_corr_methods import (load_attentions, MaxCorr, MinCorr,
                                    PearsonMaxCorr, PearsonMinCorr,
                                    JSMaxCorr, JSMinCorr, AttnLinCKA,
                                    AttnCCA)

def get_options(opt_fname):
    if opt_fname == None:
        layerspec_l = [-1] * len(attention_fname_l)
    else:
        with open(opt_fname, 'r') as f:
            l = [line.strip() for line in f]
            #opt_l = [line.strip().split(',') for line in f]
            #l, f, s = zip(*opt_l)

            layerspec_l = []
            for ls in l:
                if ls == "all":
                    layerspec_l.append(ls)
                else:
                    layerspec_l.append(int(ls))

    return layerspec_l

def get_method_l(methods, num_heads_d, attentions_d, device):
    if 'all' in methods:
        method_l = [
            MaxCorr(num_heads_d, attentions_d, device),
            MinCorr(num_heads_d, attentions_d, device),
            PearsonMaxCorr(num_heads_d, attentions_d, device),
            PearsonMinCorr(num_heads_d, attentions_d, device),
            JSMaxCorr(num_heads_d, attentions_d, device),
            JSMinCorr(num_heads_d, attentions_d, device),
            AttnLinCKA(num_heads_d, attentions_d, device),
            AttnCCA(num_heads_d, attentions_d, device),
        ]
    else:
        method_l = []
        for method in methods:
            if method == 'maxcorr':
                method_l.append(MaxCorr(num_heads_d, attentions_d, device))
            elif method == 'mincorr':
                method_l.append(MinCorr(num_heads_d, attentions_d, device))
            elif method == 'pearsonmaxcorr':
                method_l.append(PearsonMaxCorr(num_heads_d,
                                               attentions_d, device))
            elif method == 'pearsonmincorr':
                method_l.append(PearsonMinCorr(num_heads_d,
                                               attentions_d, device))
            elif method == 'jsmaxcorr':
                method_l.append(JSMaxCorr(num_heads_d, attentions_d,
                                          device))
            elif method == 'jsmincorr':
                method_l.append(JSMinCorr(num_heads_d, attentions_d,
                                          device))
            elif method == 'attn_lincka':
                method_l.append(AttnLinCKA(num_heads_d, attentions_d,
                                          device))
            elif method == 'attn_cca':
                method_l.append(AttnCCA(num_heads_d, attentions_d,
                                          device))

    return method_l

def main(methods, attention_files, output_file, opt_fname=None,
         limit=None, disable_cuda=False, ar_mask=False):

    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Using device: {0}".format(device))

    # Set `attention_fname_l`, and options
    with open(attention_files) as f:
        attention_fname_l = [line.strip() for line in f]

    layerspec_l = get_options(opt_fname)
    
    # Load
    print("Loading attentions")
    a = load_attentions(attention_fname_l, limit=limit,
                        layerspec_l=layerspec_l, ar_mask=ar_mask)
    num_heads_d, attentions_d = a
    
    # Set `method_l`, list of Method objects
    print('\nInitializing methods ' + str(methods))
    method_l = get_method_l(methods, num_heads_d, attentions_d, device)

    # Run all methods in method_l
    print('\nComputing correlations')
    for method in method_l:
        print('For method: ', str(method))
        method.compute_correlations()

    print('\nWriting correlations')
    for method in method_l:
        print('For method: ', str(method))
        out_fname = output_file + '_' + str(method)
        method.write_correlations(out_fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("attention_files")
    parser.add_argument("output_file")
    parser.add_argument("--opt_fname", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--ar_mask", action="store_true") 

    args = parser.parse_args()
    main(args.methods, args.attention_files, args.output_file,
         opt_fname=args.opt_fname, limit=args.limit,
         disable_cuda=args.disable_cuda, ar_mask=args.ar_mask) 


