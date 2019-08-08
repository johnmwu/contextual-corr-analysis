"""
Compute means and stdevs. One off script. 

Usage: 
  mk_meanstd.py REPRESENTATION_FILES OUTPUT_FILE

Arguments:
  REPRESENTATION_FILES   File containing list of locations of representation files 
                        (one per line)
  OUTPUT_FILE       File to write the output results

Options:
  -h, --help                             show this help message  

"""

from docopt import docopt
import torch
import h5py
import json
from tqdm import tqdm

def main(representation_files, output_file):
    with open(representation_files) as f:
        representation_fname_l = [line.strip() for line in f]

    # Set `means_d`, `stdevs_d`
    # {fname: tensor}
    means_d = {}
    stdevs_d = {}
    for fname in tqdm(representation_fname_l, desc='mu, sigma'):
        # Create `activations_h5`, `sentence_d`, `indices`
        activations_h5 = h5py.File(fname, 'r')
        sentence_d = json.loads(activations_h5['sentence_to_index'][0])
        temp = {} 
        for k, v in sentence_d.items():
            temp[v] = k
        sentence_d = temp # {str ix, sentence}
        indices = list(sentence_d.keys())

        # Set `activations` tensor, `dim`
        dim = len(activations_h5['0'].shape)
        activations = torch.cat([torch.tensor(activations_h5[str_ix])
                                    for str_ix in sentence_d],
                                dim=0 if dim==2 else 1)

        # Update `means_d`, `stdevs_d`
        if dim == 2: 
            means_d[fname] = torch.mean(activations, dim=0, keepdim=False)
            stdevs_d[fname] = torch.std(activations, dim=0, keepdim=False,
                                        unbiased=False)
        elif dim == 3:
            means_d[fname] = torch.mean(activations, dim=1, keepdim=False)
            stdevs_d[fname] = torch.std(activations, dim=1, keepdim=False,
                                        unbiased=False)

        activations_h5.close() 

    # Write
    means_fname = output_file + "_means"
    stdevs_fname = output_file + "_stdevs"
    torch.save(means_d, means_fname)
    torch.save(stdevs_d, stdevs_fname)
    

if __name__ == "__main__":
    args = docopt(__doc__)

    main(args['REPRESENTATION_FILES'], args['OUTPUT_FILE'])
