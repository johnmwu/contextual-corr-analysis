"""
Compute max and min. One off script. 

Usage: 
  mk_maxmin.py REPRESENTATION_FILES OUTPUT_FILE

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

    # Set `max_d`, `min_d`
    # {fname: tensor}
    max_d = {}
    min_d = {}
    for fname in tqdm(representation_fname_l, desc='max, min'):
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

        # Update `max_d`, `min_d`
        if dim == 2: 
            max_d[fname] = torch.max(activations, dim=0, keepdim=False)[0]
            min_d[fname] = torch.min(activations, dim=0, keepdim=False)[0]
        elif dim == 3:
            max_d[fname] = torch.max(activations, dim=1, keepdim=False)[0]
            min_d[fname] = torch.min(activations, dim=1, keepdim=False)[0]

        activations_h5.close() 

    # Write
    max_fname = output_file + "_max"
    min_fname = output_file + "_min"
    torch.save(max_d, max_fname)
    torch.save(min_d, min_fname)
    

if __name__ == "__main__":
    args = docopt(__doc__)

    main(args['REPRESENTATION_FILES'], args['OUTPUT_FILE'])
