import torch 
from tqdm import tqdm
from itertools import product as p
import json
import numpy as np
import h5py


class Method(object):
    """
    Abstract representation of a correlation method. 

    Example instances are MaxCorr, MinCorr, LinReg, SVCCA, CKA. 
    """
    def __init__(self, representation_files, layer=None, first_half_only=False,
                 second_half_only=False):
        self.representation_files = [line.strip() for line in representation_files]
        self.first_half_only = first_half_only
        self.second_half_only = second_half_only

    def load_representations(self, limit=None):
        """
        Load data. 

        Set `self.num_neurons_d` and `self.representations_d`. 
        """

        self.num_neurons_d = {}
        self.representations_d = {}

        for fname in tqdm(representation_files, desc='loading'):
            # this (the formatting) follows contexteval:
            # https://github.com/nelson-liu/contextual-repr-analysis/blob/master/contexteval/contextualizers/precomputed_contextualizer.py
            # Create `activations_h5`, `sentence_d`, `indices`
            activations_h5 = h5py.File(fname)
            sentence_d = json.loads(activations_h5['sentence_to_index'][0])
            temp = {}
            for k, v in sentence_d.items():
                temp[v] = k
            sentence_d = temp # {str ix, sentence}
            indices = list(sentence_d.keys())[:limit]

            # Create `representations_l`
            representations_l = []
            for sentence_ix in indices: 
                # Create `activations`
                activations = torch.FloatTensor(activations_h5[sentence_ix])
                if not (activations.dim() == 2 or activations.dim() == 3):
                    raise ValueError('Improper array dimension in file: ' + fname +
                                     "\nShape: " + str(activations.shape))

                # Create `representations`
                representations = activations
                if activations.dim() == 3:
                    if self.layer is not None: 
                        representations = activations[self.layer] 
                    else:
                        # use the top layer by default
                        representations = activations[-1]
                if self.first_half_only: 
                    representations = torch.chunk(representations, chunks=2, dim=-1)[0]
                elif self.second_half_only:
                    representations = torch.chunk(representations, chunks=2, dim=-1)[1]

                representations_l.append(representations)

            self.num_neurons_d[fname] = representations_l[0].size()[1]
            self.representations_d[fname] = torch.cat(representations_l) # TO DO: .cpu()?

    def compute_correlations(self):
        raise NotImplementedError

    def write_correlations(self):
        raise NotImplementedError


class MaxMinCorr(Method):
    def __init__(self, representation_files):
        super().__init__(representation_files)

    def compute_correlations(self, op):
        """
        Set `self.correlations`, `self.clusters`, `self.neuron_sort`. 
        """

        # Set `means_d`, `stdevs_d`
        means_d = {}
        stdevs_d = {}
        for network in tqdm(self.representations_d, desc='mu, sigma'):
            t = self.representations_d[network]

            means_d[network] = t.mean(0, keepdim=True)
            stdevs_d[network] = (t - means_d[network]).pow(2).mean(0, keepdim=True).pow(0.5)

        # Set `self.correlations`
        # {network: {other: tensor}}
        self.correlations = {network: {} for network in self.representations_d}
        num_words = list(self.representations_d.values())[0].size()[0] # TO DO: make more elegant

        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                           desc='correlate',
                                           total=len(self.representations_d)**2):
            if network == other_network:
                continue

            if other_network in self.correlations[network].keys(): # TO DO: optimize?
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
                                key = lambda i: abs(self.correlations[network][other][neuron][i])) 
                     for other in self.correlations[network]
                }

        # Set `self.neuron_sort`
        # {network, sorted_list}
        self.neuron_sort = {} 
        # Sort neurons by worst (or best) best correlation with another neuron
        # in another network.
        for network in tqdm(self.representations_d, desc='annotation'):
            self.neuron_sort[network] = sorted(
                range(self.num_neurons_d[network]),
                key = lambda i : op(
                    abs(self.correlations[network][other][i][self.clusters[network][i][other]])
                    for other in self.clusters[network][i]),
                reverse=True
            )


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
    # TO DO: test (I don't think it's wrong, though)
    def __init__(self, representation_files):
        super().__init__(representation_files)

    def compute_correlations(self):
        super().compute_correlations(max)


class MinCorr(MaxMinCorr):
    # TO DO: test (I don't think it's wrong, though)
    def __init__(self, representation_files):
        super().__init__(representation_files)

    def compute_correlations(self):
        super().compute_correlations(min)


class LinReg(Method):
    def __init__(self, representation_files):
        super().__init__(representation_files)

    def compute_correlations(self):
        """
        Set `self.neuron_sort`. 
        """

        # Set `means_d`, `stdevs_d`, normalize to mean 0 std 1
        # Not exactly sure why we compute `means_d`
        means_d = {}
        stdevs_d = {}

        for network in tqdm(self.representations_d, desc='mu, sigma'):
            t = self.representations_d[network]
            means = t.mean(0, keepdim=True)
            stdevs = t.std(0, keepdim=True)

            means_d[network] = means
            stdevs_d[network] = stdevs
            self.representations_d[network] = (t - means) / stdevs

        # Set `self.errors`
        # {network: {other: error_tensor}}
        self.errors = {network: {} for network in self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d), desc='correlate',
                                           total=len(self.representations_d)**2):
            if network == other_network:
                continue

            # Try to predict this network given the other one
            X = self.representations_d[other_network].cpu().numpy()
            Y = self.representations_d[network].cpu().numpy()

            # solve with ordinary least squares 
            error = np.linalg.lstsq(X, Y, rcond=None)[1] # TO DO: don't use numpy, or at least use CUDA
            # Possibilities are use torch (torch.svd or smth), or use another library (cupy)

            # Note: what was here previously is very numerically
            # unstable. Linear regression should be performed using either QR or
            # the SVD (which are numerically stable computations). 
            if len(error) == 0:
                raise ValueError
            error = torch.from_numpy(error)

            self.errors[network][other_network] = error

        # Set `self.neuron_sort`
        # {network: sorted_list}
        self.neuron_sort = {}
        # Sort neurons by worst correlation (highest regression error) with another network
        for network in tqdm(self.representations_d, desc='annotation'):
            self.neuron_sort[network] = sorted(
                range(self.num_neurons_d[network]),
                key = lambda i: max(
                    self.errors[network][other][i] 
                    for other in self.errors[network]
                )
            )

    def write_correlations(self, output_file):
        self.neuron_notated_sort = {}
        for network in tqdm(self.representations_d, desc='write'):
            self.neuron_notated_sort[network] = [
                (
                    neuron,
                    {
                        other: float(self.errors[network][other][neuron])
                        for other in self.errors[network]
                    }
                )
                for neuron in self.neuron_sort[network]
            ]

        json.dump(self.neuron_notated_sort, open(output_file, 'w'), indent=4)


class SVCCA(Method):
    def __init__(self, representation_files, percent_variance=0.99, normalize_dimensions=False):
        super().__init__(representation_files)

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
        whitening_transforms = {} # {network: whitening_tensor}
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

        # Set `self.transforms` to be {network: {other: svcca_transform}}
        self.transforms = {network: {} for network in self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d), desc='cca',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            if other_network in self.transforms[network].keys(): # TO DO: optimize?
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


# https://debug-ml-iclr2019.github.io/cameraready/DebugML-19_paper_9.pdf
class CKA(Method):
    def __init__(self, representation_files, normalize_dimensions=True):
        super().__init__(representation_files)
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
        self.similarities = {network: {} for network in self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d), desc='cka',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            if other_network in self.similarities[network].keys(): # TO DO: optimize?
                continue

            X = self.representations_d[network]
            Y = self.representations_d[other_network]

            XtX_F = torch.norm(torch.mm(X.t(), X), p='fro').item()
            YtY_F = torch.norm(torch.mm(Y.t(), Y), p='fro').item()
            YtX_F = torch.norm(torch.mm(Y.t(), X), p='fro').item()

            # eq 5 in paper
            self.similarities[network][other_network] = YtX_F**2 / (XtX_F*YtY_F)

    def write_correlations(self, output_file):
        torch.save(self.similarities, output_file)

