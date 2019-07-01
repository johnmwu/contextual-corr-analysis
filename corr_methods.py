import torch 
from tqdm import tqdm
from itertools import product as p
import json
import numpy as np
import h5py

def load_representations(representation_fname_l, limit=None, layer=None,
                         first_half_only=False, second_half_only=False,
                         disable_cuda=False):
    """
    Load data. Returns `num_neurons_d` and `representations_d`. 

    Parameters
    ----
    representation_fname_l : list<str>
        List of filenames. 
    limit : int or NoneType
        Cap on the number of data points. None if no cap. 
    layer : int or NoneType
        Layer to correlate. None if the top layer. 

        Currently (d1c0249), you are forced to always correlate the same
        layer of each model.
    first_half_only : bool
        Only use the first half of the neurons.
    second_half_only : bool
        Only use the second half of the neurons. 
    disable_cuda : bool
        Disable CUDA. 

    Returns
    ----
    num_neurons_d : dict<str, int>
        Dict of {fname : num_neurons}
    representations_d : dict<str, tensor>
        Dict of {fname : (len_data, num_neurons) tensor}. The tensor
        contains the activations for a given model on each input data
        point. 
    """
    num_neurons_d = {} 
    representations_d = {} 

    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # formatting follows contexteval:
    # https://github.com/nelson-liu/contextual-repr-analysis/blob/master/contexteval/contextualizers/precomputed_contextualizer.py
    for fname in tqdm(representation_fname_l, desc='loading'):
        # Create `activations_h5`, `sentence_d`, `indices`
        activations_h5 = h5py.File(fname, 'r')
        sentence_d = json.loads(activations_h5['sentence_to_index'][0])
        temp = {} # TO DO: Make this more elegant?
        for k, v in sentence_d.items():
            temp[v] = k
        sentence_d = temp # {str ix, sentence}
        indices = list(sentence_d.keys())[:limit]

        # Create `representations_l`
        representations_l = []
        for sentence_ix in indices: 
            # Create `activations`
            activations = torch.FloatTensor(activations_h5[sentence_ix])
            activations = activations.to(device)
                                            
            if not (activations.dim() == 2 or activations.dim() == 3):
                raise ValueError('Improper array dimension in file: ' +
                                 fname + "\nShape: " +
                                 str(activations.shape))

            # Create `representations`
            representations = activations
            if activations.dim() == 3:
                if layer is not None: 
                    representations = activations[layer] 
                else:
                    # use the top layer by default
                    representations = activations[-1]
            if first_half_only: 
                representations = torch.chunk(representations, chunks=2,
                                              dim=-1)[0]
            elif second_half_only:
                representations = torch.chunk(representations, chunks=2,
                                              dim=-1)[1]

            representations_l.append(representations)

        num_neurons_d[fname] = representations_l[0].size()[1]
        representations_d[fname] = torch.cat(representations_l) 

    return (num_neurons_d, representations_d)


class Method(object):
    """
    Abstract representation of a correlation method. 

    Example instances are MaxCorr, MinCorr, LinReg, SVCCA, CKA. 
    """
    def __init__(self, num_neurons_d, representations_d):
        self.num_neurons_d = num_neurons_d
        self.representations_d = representations_d

    def compute_correlations(self):
        raise NotImplementedError

    def write_correlations(self):
        raise NotImplementedError


class MaxMinCorr(Method):
    def __init__(self, num_neurons_d, representations_d, op=None):
        super().__init__(num_neurons_d, representations_d)
        self.op = op

    def compute_correlations(self):
        """
        Set `self.correlations`, `self.clusters`, `self.neuron_sort`. 
        """

        if self.op is None:
            raise ValueError('self.op not set in MaxMinCorr')

        # Set `means_d`, `stdevs_d`
        means_d = {}
        stdevs_d = {}
        for network in tqdm(self.representations_d, desc='mu, sigma'):
            t = self.representations_d[network]

            means_d[network] = t.mean(0, keepdim=True)
            stdevs_d[network] = (t - means_d[network]).pow(2).mean(0, keepdim=True).pow(0.5)

        # Set `self.correlations`
        # {network: {other: tensor}}
        self.correlations = {network: {} for network in
                             self.representations_d}
        num_words = next(iter(self.representations_d.values())).size()[0]

        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                             desc='correlate',
                                             total=len(self.representations_d)**2):
            if network == other_network:
                continue

            if other_network in self.correlations[network]: 
                continue

            t1 = self.representations_d[network] # "tensor"
            t2 = self.representations_d[other_network] 
            m1 = means_d[network] # "means"
            m2 = means_d[other_network]
            s1 = stdevs_d[network] # "stdevs"
            s2 = stdevs_d[other_network]

            covariance = (torch.mm(t1.t(), t2) / num_words # E[ab]
                          - torch.mm(m1.t(), m2)) # E[a]E[b]
            correlation = covariance / torch.mm(s1.t(), s2)

            correlation = correlation.cpu().numpy()
            self.correlations[network][other_network] = correlation
            self.correlations[other_network][network] = correlation.T

        # Set `self.clusters`
        # {network: {neuron: {other: other_neuron}}}
        self.clusters = {network: {} for network in self.representations_d} 
        for network in tqdm(self.representations_d, desc='self.clusters',
                            total=len(self.representations_d)):
            for neuron in range(self.num_neurons_d[network]): 
                self.clusters[network][neuron] = {
                    other : max(range(self.num_neurons_d[other]),
                                key=lambda i:
                                abs(self.correlations[network][other][neuron][i]))
                    for other in self.correlations[network]
                }

        # Set `self.neuron_sort`
        # {network, sorted_list}
        self.neuron_sort = {} 
        for network in tqdm(self.representations_d, desc='annotation'):
            self.neuron_sort[network] = sorted(
                range(self.num_neurons_d[network]), key=lambda i :
                self.op(
                    abs(self.correlations[network][other][i][self.clusters[network][i][other]])
                    for other in self.clusters[network][i]), reverse=True )


    def write_correlations(self, output_file):
        """
        Create `self.neuron_notated_sort`, and write it to output_file. 
        """

        self.neuron_notated_sort = {}
        for network in tqdm(self.representations_d, desc='write'):
            self.neuron_notated_sort[network] = [
                    (
                        neuron, 
                        {
                            '%s:%d' % (other, self.clusters[network][neuron][other],):
                            float(self.correlations[network][other][neuron][self.clusters[network][neuron][other]])
                            for other in self.clusters[network][neuron]
                        }
                    )
                    for neuron in self.neuron_sort[network]
                ]

        json.dump(self.neuron_notated_sort, open(output_file, "w"), indent=4)


class MaxCorr(MaxMinCorr):
    def __init__(self, num_neurons_d, representations_d):
        super().__init__(num_neurons_d, representations_d, op=max)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "maxcorr"


class MinCorr(MaxMinCorr):
    def __init__(self, num_neurons_d, representations_d):
        super().__init__(num_neurons_d, representations_d, op=min)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "mincorr"


class LinReg(Method): 
    def __init__(self, num_neurons_d, representations_d):
        super().__init__(num_neurons_d, representations_d)

    def compute_correlations(self):
        """
        Set `self.neuron_sort`. 
        """
        # Set `means_d`, `stdevs_d`, normalize to mean 0 std 1
        means_d = {}
        stdevs_d = {}

        for network in tqdm(self.representations_d, desc='mu, sigma'):
            t = self.representations_d[network]
            means = t.mean(0, keepdim=True)
            stdevs = (t - means).pow(2).mean(0, keepdim=True).pow(0.5)

            means_d[network] = means
            stdevs_d[network] = stdevs
            self.representations_d[network] = (t - means) / stdevs

        # Set `self.pred_power`
        # If the data is centered, it is the r value. 
        self.pred_power = {network: {} for network in self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                           desc='correlate',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            X = self.representations_d[other_network]
            Y = self.representations_d[network]

            # SVD method of linreg
            U, S, V = torch.svd(X) 
            UtY = torch.mm(U.t(), Y) # b for Ub = Y

            bnorms = torch.norm(UtY, dim=0)
            ynorms = torch.norm(Y, dim=0)

            self.pred_power[network][other_network] = bnorms / ynorms

        # Set `self.neuron_sort`
        # {network: sorted_list}
        self.neuron_sort = {}
        # Sort neurons by worst correlation with another network
        for network in tqdm(self.representations_d, desc='annotation'):
            self.neuron_sort[network] = sorted(
                    range(self.num_neurons_d[network]),
                    key = lambda i: min(
                        self.pred_power[network][other][i] 
                        for other in self.pred_power[network]),
                    reverse=True
                )


    def write_correlations(self, output_file):
        self.neuron_notated_sort = {}
        for network in tqdm(self.representations_d, desc='write'):
            self.neuron_notated_sort[network] = [
                (
                    neuron,
                    {
                        other: float(self.pred_power[network][other][neuron])
                        for other in self.pred_power[network]
                    }
                )
                for neuron in self.neuron_sort[network]
            ]

        json.dump(self.neuron_notated_sort, open(output_file, 'w'), indent=4)

    def __str__(self):
        return "linreg"

class SVCCA(Method):
    def __init__(self, num_neurons_d, representations_d, percent_variance=0.99,
                 normalize_dimensions=False):
        super().__init__(num_neurons_d, representations_d)

        self.percent_variance = percent_variance
        self.normalize_dimensions = normalize_dimensions

    def compute_correlations(self):
        """
        Set `self.transforms` to be the svcca transform matrix M. 

        If X is the activation tensor, then X M is the svcca tensor. 
        """ 
        # Normalize
        if self.normalize_dimensions:
            for network in tqdm(self.representations_d, desc='mu, sigma'):
                t = self.representations_d[network]
                means = t.mean(0, keepdim=True)
                stdevs = t.std(0, keepdim=True)

                self.representations_d[network] = (t - means) / stdevs

        # Set `whitening_transforms`, `pca_directions`
        # {network: whitening_tensor}
        whitening_transforms = {} 
        pca_directions = {} 
        for network in tqdm(self.representations_d, desc='pca'):
            X = self.representations_d[network]
            U, S, V = torch.svd(X)

            var_sums = torch.cumsum(S.pow(2), 0)
            wanted_size = torch.sum(var_sums.lt(var_sums[-1] * self.percent_variance)).item()

            print('For network', network, 'wanted size is', wanted_size)

            whitening_transform = torch.mm(V, torch.diag(1/S))
            whitening_transforms[network] = whitening_transform[:, :wanted_size]
            pca_directions[network] = U[:, :wanted_size]

        # Set `self.transforms`
        # {network: {other: svcca_transform}}
        self.transforms = {network: {} for network in self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                           desc='cca',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            if other_network in self.transforms[network]: 
                continue

            X = pca_directions[network]
            Y = pca_directions[other_network]

            # Perform SVD for CCA.
            # u s vt = Xt Y
            # s = ut Xt Y v
            u, s, v = torch.svd(torch.mm(X.t(), Y))

            self.transforms[network][other_network] = torch.mm(whitening_transforms[network], u)
            self.transforms[other_network][network] = torch.mm(whitening_transforms[other_network], v)

    def write_correlations(self, output_file):
        torch.save(self.transforms, output_file)

    def __str__(self):
        return "svcca"

# https://debug-ml-iclr2019.github.io/cameraready/DebugML-19_paper_9.pdf
class CKA(Method):
    def __init__(self, num_neurons_d, representations_d,
                 normalize_dimensions=True):
        super().__init__(num_neurons_d, representations_d)
        self.normalize_dimensions = normalize_dimensions

    def compute_correlations(self):
        """
        Set `self.similarities`. 
        """
        # Normalize
        if self.normalize_dimensions:
            for network in tqdm(self.representations_d, desc='mu, sigma'):
                # TODO: might not need to normalize, only center
                t = self.representations_d[network]
                means = t.mean(0, keepdim=True)
                stdevs = t.std(0, keepdim=True)

                self.representations_d[network] = (t - means) / stdevs

        # Set `self.similarities`
        # {network: {other: cka_similarity}}
        self.similarities = {network: {} for network in
                             self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                           desc='cka',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            if other_network in self.similarities[network]: 
                continue

            X = self.representations_d[network]
            Y = self.representations_d[other_network]

            XtX_F = torch.norm(torch.mm(X.t(), X), p='fro').item()
            YtY_F = torch.norm(torch.mm(Y.t(), Y), p='fro').item()
            YtX_F = torch.norm(torch.mm(Y.t(), X), p='fro').item()

            # eq 5 in paper
            self.similarities[network][other_network] = (YtX_F**2 /
                                                         (XtX_F*YtY_F))

    def write_correlations(self, output_file):
        torch.save(self.similarities, output_file)

    def __str__(self):
        return "cka"
        
