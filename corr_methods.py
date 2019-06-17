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
    def __init__(self, embedding_files, layer=None, first_half_only=False,
                 second_half_only=False):
        self.embedding_files = [line.strip() for line in embedding_files]
        self.first_half_only = first_half_only
        self.second_half_only = second_half_only

    def load_embeddings(self, limit=None):
        self.num_neurons_d = {}
        self.embeddings_d = {}

        for fname in tqdm(embedding_files, desc='loading'):
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

            # Create `embeddings_l`
            embeddings_l = []
            for sentence_ix in indices: 
                # Create `activations`
                activations = torch.FloatTensor(activations_h5[sentence_ix])
                if not (activations.dim() == 2 or activations.dim() == 3):
                    raise ValueError('Improper array shape in file: ' + fname +
                                     "\nShape: " + str(activations.shape))

                # Create `embeddings`
                embeddings = activations
                if activations.dim() == 3:
                    if self.layer is not None: 
                        embeddings = activations[self.layer] 
                    else:
                        # use the top layer by default
                        embeddings = activations[-1]
                if self.first_half_only: 
                    embeddings = torch.chunk(embeddings, chunks=2, dim=-1)[0]
                elif self.second_half_only:
                    embeddings = torch.chunk(embeddings, chunks=2, dim=-1)[1]

                embeddings_l.append(embeddings)

            self.num_neurons_d[fname] = embeddings_l[0].size(1)
            self.embeddings_d[fname] = torch.cat(embeddings_l)

    def compute_correlations(self):
        raise NotImplementedError

    def write_correlations(self):
        raise NotImplementedError


class MaxMinCorr(Method):

    def __init__(self, embedding_files, op):
        super().__init__(embedding_files)


    def compute_correlations(self, op):

        # Get means and stdevs so that we can whiten appropriately
        means = {}
        stdevs = {}
        for network in tqdm(self.activations, desc='mu, sigma'):
            means[network] = self.activations[network].mean(0, keepdim=True)
            stdevs[network] = (
                self.activations[network] - means[network].expand_as(self.activations[network])
            ).pow(2).mean(0, keepdim=True).pow(0.5)

        self.correlations = {network: {} for network in self.activations}

        # Get all correlation pairs
        for network, other_network in tqdm(p(self.activations, self.activations), desc='correlate', total=len(self.activations)**2):
            # TODO: relax this, but be careful about the max (it will be self)
            # Don't match within one network
            if network == other_network:
                continue

            # Correlate these networks with each other
            covariance = (
                torch.mm(
                    self.activations[network].t(), self.activations[other_network] # E[ab]
                ) / self.activations[network].size()[0]
                - torch.mm(
                    means[network].t(), means[other_network] # E[a]E[b]
                )
            )

            correlation = covariance / torch.mm(
                stdevs[network].t(), stdevs[other_network]
            )

            self.correlations[network][other_network] = correlation.cpu().numpy()

        # Get all "best correlation pairs"
        self.clusters = {network: {} for network in self.activations}
        for network, neuron in tqdm(self.activations, desc='clusters', total=len(self.activations)):
            for neuron in tqdm(range(self.num_neurons[network])): 
                self.clusters[network][neuron] = {
                    other: max(
                        range(self.num_neurons[network]),
                        key = lambda i: abs(self.correlations[network][other][neuron][i])
                    ) for other in self.correlations[network]
                }

        self.neuron_sort = {}    
        # Sort neurons by worst (or best) best correlation with another neuron
        # in another network.
        for network in tqdm(self.activations, desc='annotation'):
            self.neuron_sort[network] = sorted(
                range(self.num_neurons[network]),
                key = lambda i: -op(
                    abs(self.correlations[network][other][i][self.clusters[network][i][other]])
                    for other in self.clusters[network][i]
                )
            )              


    def write_correlations(self, output_file):

        # For each network, created an "annotated sort"
        self.neuron_notated_sort = {}
        for network in tqdm(self.activations, desc='write'):
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
        json.dump(neuron_notated_sort, open(output_file, 'w'), indent = 4)


class MaxCorr(MaxMinCorr):

    def __init__(self, embedding_files):

        super().__init__(embedding_files, max)


class MinCorr(MaxMinCorr):

    def __init__(self, embedding_files):

        super().__init__(embedding_files, min)


class LinReg(Method):

    def __init__(self, embedding_files):

        super().__init__(embedding_files)


    def compute_correlations(self):

        # Normalize to have mean 0 and standard devaition 1.
        means = {}
        stdevs = {}
        for network in tqdm(self.activations, desc='mu, sigma'):
            means[network] = self.activations[network].mean(0, keepdim=True)
            stdevs[network] = (
                self.activations[network] - means[network].expand_as(self.activations[network])
            ).pow(2).mean(0, keepdim=True).pow(0.5)

            self.activations[network] = (self.activations[network] - means[network]) / stdevs[network]

        self.errors = {network: {} for network in self.activations}

        # Get all correlation pairs
        for network, other_network in tqdm(p(self.activations, self.activations), desc='correlate', total=len(self.activations)**2):
            # TODO: relax this, but be careful about the max (it will be self)
            # Don't match within one network
            if network == other_network:
                continue

            # Try to predict this network given the other one
            X = self.activations[other_network].clone()
            Y = self.activations[network].clone()

            # solve with ordinary least squares 
            coefs = X.t().mm(X).inverse().mm(X.t()).mm(Y)
            prediction = X.mm(coefs)
            error = (prediction - Y).pow(2).mean(0).squeeze()

            self.errors[network][other_network] = error

        
        self.neuron_sort = {}
        # Sort neurons by best correlation (lowest regression error) with another network
        # in another network.
        for network in tqdm(self.activations, desc='annotation'):
            self.neuron_sort[network] = sorted(
                range(self.num_neurons[network]),
                key = lambda i: max(
                    self.errors[network][other][i]
                    for other in self.errors[network] if other != 'position'
                )
            )



    def write_correlations(self, output_file):

        self.neuron_notated_sort = {}
        # For each network, created an "annotated sort"
        for network in tqdm(self.activations, desc='write'):
            # Annotate each neuron with its associated cluster
            self.neuron_notated_sort[network] = [
                (
                    neuron,
                    {
                        other: self.errors[network][other][neuron]
                        for other in self.errors[network]
                    }
                )
                for neuron in self.neuron_sort[network]
            ]

        json.dump(self.neuron_notated_sort, open(output_file, 'w'), indent = 4)


class SVCCA(Method):

    def __init__(self, embedding_files, percent_variance=0.99, normalize_dimensions=False):

        super().__init__(embedding_files)

        # Percentage of variance to take in initial PCA
        self.percent_variance = percent_variance
        # Normalize dimensions first
        self.normalize_dimensions = normalize_dimensions


    def compute_correlations(self):

        # Whiten dimensions
        if self.normalize_dimensions:
            for network in tqdm(self.activations, desc='mu, sigma'):
                self.activations[network] -= self.activations[network].mean(0)
                self.activations[network] /= self.activations[network].std(0)

        # PCA to get independent components
        whitening_transforms = {}
        for network in tqdm(self.activations, desc='pca'):
            X = self.activations[network]
            covariance = torch.mm(X.t(), X) / (X.size()[0] - 1)

            e, v = torch.eig(covariance, eigenvectors = True)

            # Sort by eigenvector magnitude
            magnitudes, indices = torch.sort(torch.abs(e[:, 0]), dim = 0, descending = True)
            se, sv = e[:, 0][indices], v.t()[indices].t()

            # Figure out how many dimensions account for 99% of the variance
            var_sums = torch.cumsum(se, 0)
            wanted_size = torch.sum(var_sums.lt(var_sums[-1] * args.percent_variance))

            print('For network', network, 'wanted size is', wanted_size)

            # This matrix has size (dim) x (dim)
            whitening_transform = torch.mm(sv, torch.diag(se ** -0.5))

            # We wish to cut it down to (dim) x (wanted_size)
            whitening_transforms[network] = whitening_transform[:, :wanted_size]

            #print(covariance[:10, :10])
            #print(torch.mm(whitening_transforms[network], whitening_transforms[network].t())[:10, :10])

        # CCA to get shared space
        self.transforms = {}
        for a, b in tqdm(p(self.activations, self.activations), desc = 'cca', total = len(self.activations) ** 2):
            if a is b or (a, b) in self.transforms or (b, a) in self.transforms:
                continue

            X, Y = self.activations[a], self.activations[b]

            # Apply PCA transforms to get independent things
            X = torch.mm(X, whitening_transforms[a])
            Y = torch.mm(Y, whitening_transforms[b])

            # Get a correlation matrix
            correlation_matrix = torch.mm(X.t(), Y) / (X.size()[0] - 1)

            # Perform SVD for CCA.
            # u s vt = Xt Y
            # s = ut Xt Y v
            u, s, v = torch.svd(correlation_matrix)

            X = torch.mm(X, u).cpu()
            Y = torch.mm(Y, v).cpu()

            self.transforms[a, b] = {
                a: whitening_transforms[a].mm(u),
                b: whitening_transforms[b].mm(v)
            }


    def write_correlations(self, output_file):

        torch.save(self.transforms, output_file)


# https://debug-ml-iclr2019.github.io/cameraready/DebugML-19_paper_9.pdf
class CKA(Method):

    def __init__(self, embedding_files, normalize_dimensions=True):

        super().__init__(embedding_files)
        self.normalize_dimensions = normalize_dimensions


    def compute_correlations(self):

        # Whiten dimensions
        if self.normalize_dimensions:
            for network in tqdm(self.activations, desc='mu, sigma'):
                self.activations[network] -= self.activations[network].mean(0)
                # TODO: might not need to normalize, only center
                self.activations[network] /= self.activations[network].std(0)

        # CKA to get shared space
        self.similarities = {}
        for a, b in tqdm(p(self.activations, self.activations), desc = 'cca', total = len(self.activations) ** 2):
            if a is b or (a, b) in self.transforms or (b, a) in self.transforms:
                continue

            X, Y = self.activations[a], self.activations[b]        

            self.similarities[a, b] = torch.norm(torch.mm(Y.t(), X), p='fro').pow(2) / ( 
                torch.norm(torch.mm(X.t(), X), p='fro') * torch.norm(torch.mm(Y.to(), Y), p='fro')
                )


    def write_correlations(self, output_file):

        torch.save(self.transforms, output_file)

