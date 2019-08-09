import argparse
import torch
from tqdm import tqdm
import pickle
import os.path
from os.path import basename, dirname
import h5py
import json

representation_fname_l = [
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/bert_large_cased/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/openai_transformer/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/bert_base_cased/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/elmo_original/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/calypso_transformer_6_512_base/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/elmo_4x4096_512/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlnet_large_cased/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/gpt2_small/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlm-mlm-enro-1024/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlm-mlm-enfr-1024/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/gpt2_medium/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlm-mlm-en-2048/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlm-clm-ende-1024/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlnet_base_cased/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlm-mlm-ende-1024/ptb_pos_dev.hdf5',
    '/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlm-clm-enfr-1024/ptb_pos_dev.hdf5',
]    

STATISTICS = [
    "mean",
    "std",
    "max",
    "min",
]

output_fname = "/data/sls/temp/johnmwu/contextual-corr-analysis/stats"

def fname2mname(fname):
    """
    "filename to model name". 
    """
    return basename(dirname(fname))

def main(overwrite=False):
    # Functionality for if the tensors are rank 2 has not been added

    # Initialize `stats`
    if overwrite:
        stats = {stat: {} for stat in STATISTICS}
    elif not os.path.isfile(output_fname):
        stats = {stat: {} for stat in STATISTICS}
    else:
        with open(output_fname, 'rb') as f:
            stats = pickle.load(f)

    # Main loop
    for fname in representation_fname_l:
        mname = fname2mname(fname)

        # Set `stats_to_compute`
        stats_to_compute = []
        for stat in STATISTICS:
            if mname not in stats[stat]:
                stats_to_compute.append(stat)

        if len(stats_to_compute) == 0: # nothing to do
            continue

        # Set `representations`
        activations_h5 = h5py.File(fname, 'r')
        sentence_d = json.loads(activations_h5['sentence_to_index'][0])
        temp = {} # TO DO: Make this more elegant?
        for k, v in sentence_d.items():
            temp[v] = k
        sentence_d = temp # {str ix, sentence}
        indices = list(sentence_d.keys())

        representations_l = []
        for sentence_ix in indices:
            activations = torch.FloatTensor(activations_h5[sentence_ix])
            representations_l.append(activations)

        representations = torch.cat(representations_l, dim=1)
        activations_h5.close()

        # Update `stats`
        for stat in stats_to_compute:
            if stat == "max":
                stats["max"][mname] = torch.max(activations, dim=1)[0].cpu().numpy()
            elif stat == "min":
                stats["min"][mname] = torch.min(activations, dim=1)[0].cpu().numpy()
            elif stat == "mean":
                stats["mean"][mname] = torch.mean(activations, dim=1).cpu().numpy()
            elif stat == "std":
                stats["std"][mname] = torch.std(activations, dim=1).cpu().numpy()

    # Write `stats`
    with open(output_fname, 'wb') as f:
        pickle.dump(stats, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(overwrite=args.overwrite)
