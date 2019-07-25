import torch 
from tqdm import tqdm
from itertools import product as p
import json
import numpy as np
import h5py
from os.path import basename, dirname
import dask.array as da

def load_representations(representation_fname_l, limit=None,
                         layerspec_l=None, first_half_only_l=False,
                         second_half_only_l=False):
    """
    Load data. Returns `num_neurons_d` and `representations_d`. 

    Parameters
    ----
    representation_fname_l : list<str>
        List of filenames. 
    limit : int or None
        Cap on the number of data points (here, sentences). None if no cap.
    layer : TO DO
        Layer to correlate. None if the top layer. 

        Currently (d1c0249), you are forced to always correlate the same
        layer of each model.
    first_half_only : TO DO
        Only use the first half of the neurons.
    second_half_only : TO DO
        Only use the second half of the neurons. 

    Returns
    ----
    num_neurons_d : dict<str, int>
        Dict of {repr_name : num_neurons}
    representations_d : dict<str, tensor>
        Dict of {repr_name : (len_data, num_neurons) tensor}. The tensor
        contains the activations for a given model on each input data
        point. 
    """
    def fname2mname(fname):
        """
        "filename to model name". 
        """
        return basename(dirname(fname))

    num_neurons_d = {} 
    representations_d = {} 

    # formatting follows contexteval:
    # https://github.com/nelson-liu/contextual-repr-analysis/blob/master/contexteval/contextualizers/precomputed_contextualizer.py
    for loop_var in tqdm(zip(representation_fname_l, layerspec_l,
                             first_half_only_l, second_half_only_l)):
        fname, layerspec, first_half_only, second_half_only = loop_var

        # Set `activations_h5`, `sentence_d`, `indices`
        activations_h5 = h5py.File(fname, 'r')
        sentence_d = json.loads(activations_h5['sentence_to_index'][0])
        temp = {} # TO DO: Make this more elegant?
        for k, v in sentence_d.items():
            temp[v] = k
        sentence_d = temp # {str ix, sentence}
        indices = list(sentence_d.keys())[:limit]

        # Set `num_layers`, `num_neurons`, `layers`
        s = activations_h5[indices[0]].shape
        num_layers = 1 if len(s)==2 else s[0]
        num_neurons = s[-1]
        if layerspec == "all":
            layers = list(range(num_layers))
        elif layerspec == "full":
            layers = ["full"]
        else:
            layers = [layerspec]

        # Set `num_neurons_d`, `representations_d`
        for layer in layers:
            # Create `representations_l`
            representations_l = []
            for sentence_ix in indices: 
                # Set `dim`
                dim = len(activations_h5[sentence_ix].shape)
                if not (dim == 2 or dim == 3):
                    raise ValueError('Improper array dimension in file: ' +
                                     fname + "\nShape: " +
                                     str(activations_h5[sentence_ix].shape))

                # Create `activations`
                if layer == "full":
                    activations = torch.FloatTensor(activations_h5[sentence_ix])
                    if dim == 3:
                        activations = activations.permute(1, 0, 2)
                        nword = activations.size()[0]
                        activations = activations.contiguous().view(nword, -1)
                else:
                    activations = torch.FloatTensor(activations_h5[sentence_ix][layer] if dim==3 
                                                        else activations_h5[sentence_ix])

                # Create `representations`
                representations = activations
                if first_half_only: 
                    representations = torch.chunk(representations, chunks=2,
                                                  dim=-1)[0]
                elif second_half_only:
                    representations = torch.chunk(representations, chunks=2,
                                                  dim=-1)[1]

                representations_l.append(representations)

            # update
            model_name = "{model}_{layer}".format(model=fname2mname(fname), 
                                                  layer=layer)
            num_neurons_d[model_name] = representations_l[0].size()[-1]
            representations_d[model_name] = torch.cat(representations_l)

    return (num_neurons_d, representations_d)


class Method(object):
    """Abstract representation of a correlation method. 

    Example instances are MaxCorr, MinCorr, MaxLinReg, MinLinReg, CCA,
    LinCKA.
    """
    def __init__(self, num_neurons_d, representations_d, device=None):
        self.num_neurons_d = num_neurons_d
        self.representations_d = representations_d
        self.device = device

    def compute_correlations(self):
        raise NotImplementedError

    def write_correlations(self):
        raise NotImplementedError


class MaxMinCorr(Method):
    def __init__(self, num_neurons_d, representations_d, device, op=None):
        super().__init__(num_neurons_d, representations_d, device)
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

            device = self.device

            t1 = self.representations_d[network].to(device) # "tensor"
            t2 = self.representations_d[other_network].to(device)
            m1 = means_d[network].to(device) # "means"
            m2 = means_d[other_network].to(device)
            s1 = stdevs_d[network].to(device) # "stdevs"
            s2 = stdevs_d[other_network].to(device)

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
    def __init__(self, num_neurons_d, representations_d, device):
        super().__init__(num_neurons_d, representations_d, device, op=max)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "maxcorr"


class MinCorr(MaxMinCorr):
    def __init__(self, num_neurons_d, representations_d, device):
        super().__init__(num_neurons_d, representations_d, device, op=min)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "mincorr"


class LinReg(Method): 
    def __init__(self, num_neurons_d, representations_d, device=None, op=None):
        super().__init__(num_neurons_d, representations_d, device)
        self.op = op

    def compute_correlations(self):
        """
        Set `self.neuron_sort`. 
        """
        # Set `means_d`, `stdevs_d`
        # Set `self.nrepresentations_d` to be normalized. 
        means_d = {}
        stdevs_d = {}
        self.nrepresentations_d = {}

        for network in tqdm(self.representations_d, desc='mu, sigma'):
            t = self.representations_d[network]
            means = t.mean(0, keepdim=True)
            stdevs = (t - means).pow(2).mean(0, keepdim=True).pow(0.5)

            means_d[network] = means
            stdevs_d[network] = stdevs
            self.nrepresentations_d[network] = (t - means) / stdevs

        # Set `self.pred_power`
        # If the data is centered, it is the r value. 
        self.pred_power = {network: {} for network in self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                           desc='correlate',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            X = self.nrepresentations_d[other_network].to(self.device)
            Y = self.nrepresentations_d[network].to(self.device)

            # SVD method of linreg
            U, S, V = torch.svd(X) 
            UtY = torch.mm(U.t(), Y) # b for Ub = Y

            bnorms = torch.norm(UtY, dim=0)
            ynorms = torch.norm(Y, dim=0)

            self.pred_power[network][other_network] = (bnorms / ynorms).cpu()

        # Set `self.neuron_sort`
        # {network: sorted_list}
        self.neuron_sort = {}
        # Sort neurons by worst correlation with another network
        for network in tqdm(self.nrepresentations_d, desc='annotation'):
            self.neuron_sort[network] = sorted(
                    range(self.num_neurons_d[network]),
                    key = lambda i: self.op(
                        self.pred_power[network][other][i] 
                        for other in self.pred_power[network]),
                    reverse=True
                )


    def write_correlations(self, output_file):
        self.neuron_notated_sort = {}
        for network in tqdm(self.nrepresentations_d, desc='write'):
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


class MaxLinReg(LinReg):
    def __init__(self, num_neurons_d, representations_d, device=None):
        super().__init__(num_neurons_d, representations_d, device, op=max)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "maxlinreg"
    

class MinLinReg(LinReg):
    def __init__(self, num_neurons_d, representations_d, device=None):
        super().__init__(num_neurons_d, representations_d, device, op=min)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "minlinreg"


class CCA(Method):
    def __init__(self, num_neurons_d, representations_d, device=None,
                 percent_variance=0.99, normalize_dimensions=True,
                 save_cca_transforms=False):
        super().__init__(num_neurons_d, representations_d, device)

        self.percent_variance = percent_variance
        self.normalize_dimensions = normalize_dimensions
        self.save_cca_transforms = save_cca_transforms

    def compute_correlations(self):
        # Normalize
        # Set `self.nrepresentations_d`
        self.nrepresentations_d = {}
        if self.normalize_dimensions:
            for network in tqdm(self.representations_d, desc='mu, sigma'):
                t = self.representations_d[network]
                means = t.mean(0, keepdim=True)
                stdevs = t.std(0, keepdim=True)

                self.nrepresentations_d[network] = (t - means) / stdevs
        else:
            self.nrepresentations_d = self.representations_d

        # Set `whitening_transforms`, `pca_directions`
        # {network: whitening_tensor}
        whitening_transforms = {} 
        pca_directions = {} 
        for network in tqdm(self.nrepresentations_d, desc='pca'):
            X = self.nrepresentations_d[network].to(self.device)
            U, S, V = torch.svd(X)

            var_sums = torch.cumsum(S.pow(2), 0)
            wanted_size = torch.sum(var_sums.lt(var_sums[-1] * self.percent_variance)).item()

            print('For network', network, 'wanted size is', wanted_size)

            whitening_transform = torch.mm(V, torch.diag(1/S))
            whitening_transforms[network] = whitening_transform[:, :wanted_size].cpu()
            pca_directions[network] = U[:, :wanted_size].cpu()

        # Set 
        # `self.transforms`: {network: {other: svcca_transform}}
        # `self.corrs`: {network: {other: canonical_corrs}}
        # `self.sv_similarities`: {network: {other: svcca_similarities}}
        # `self.pw_similarities`: {network: {other: pwcca_similarities}}
        self.transforms = {network: {} for network in self.nrepresentations_d}
        self.corrs = {network: {} for network in self.nrepresentations_d}
        self.sv_similarities = {network: {} for network in self.nrepresentations_d}
        self.pw_similarities = {network: {} for network in self.nrepresentations_d}
        for network, other_network in tqdm(p(self.nrepresentations_d,
                                             self.nrepresentations_d),
                                           desc='cca',
                                           total=len(self.nrepresentations_d)**2):

            if network == other_network:
                continue

            if other_network in self.transforms[network]: 
                continue

            X = pca_directions[network].to(self.device)
            Y = pca_directions[other_network].to(self.device)

            trans_X = whitening_transforms[network].to(self.device)
            trans_Y = whitening_transforms[other_network].to(self.device)

            # Perform SVD for CCA.
            # u s vt = Xt Y
            # s = ut Xt Y v
            u, s, v = torch.svd(torch.mm(X.t(), Y))

            # `self.transforms`, `self.corrs`, `self.sv_similarities`
            self.transforms[network][other_network] = torch.mm(trans_X, u).cpu()
            self.transforms[other_network][network] = torch.mm(trans_Y, v).cpu()

            self.corrs[network][other_network] = s.cpu()
            self.corrs[other_network][network] = s.cpu()

            self.sv_similarities[network][other_network] = s.mean().item()
            self.sv_similarities[other_network][network] = s.mean().item()

            # Compute `self.pw_similarities`. See https://arxiv.org/abs/1806.05759. 
            # This is not symmetric. 

            # For X
            H = torch.mm(X, u)
            Z = self.representations_d[network].to(self.device)
            align = torch.abs(torch.mm(H.t(), Z))
            a = torch.sum(align, dim=1, keepdim=False)
            a = a / torch.sum(a)
            self.pw_similarities[network][other_network] = torch.sum(s*a).cpu().item()

            # For Y
            H = torch.mm(Y, v)
            Z = self.representations_d[other_network].to(self.device)
            align = torch.abs(torch.mm(H.t(), Z))
            a = torch.sum(align, dim=1, keepdim=False)
            a = a / torch.sum(a)
            self.pw_similarities[other_network][network] = torch.sum(s*a).cpu().item()

    def write_correlations(self, output_file):
        if self.save_cca_transforms:
            output = {
                "transforms": self.transforms,
                "corrs": self.corrs,
                "sv_similarities": self.sv_similarities,
                "pw_similarities": self.pw_similarities,
            }
        else:
            output = {
                "corrs": self.corrs,
                "sv_similarities": self.sv_similarities,
                "pw_similarities": self.pw_similarities,
            }
        torch.save(output, output_file)

    def __str__(self):
        return "cca"

# https://arxiv.org/abs/1905.00414
class LinCKA(Method):
    def __init__(self, num_neurons_d, representations_d, device=None,
                 normalize_dimensions=True):
        # Here, normalize_dimensions means center. TODO: change. 
        super().__init__(num_neurons_d, representations_d, device)
        self.normalize_dimensions = normalize_dimensions

    def compute_correlations(self):
        def center_gram(G):
            means = G.mean(0)
            means -= means.mean() / 2
            return G - means[None, :] - means[:, None]

        def gram_rbf(X, threshold=1.0):
            if type(X) == torch.Tensor:
                dot_products = X @ X.t()
                sq_norms = dot_products.diag()
                sq_distances = -2*dot_products + sq_norms[:,None] + sq_norms[None,:]
                sq_median_distance = sq_distances.median()
                return torch.exp(-sq_distances / (2*threshold**2 * sq_median_distance))
            elif type(X) == da.Array:
                dot_products = X @ X.T
                sq_norms = da.diag(dot_products)
                sq_distances = -2*dot_products + sq_norms[:,None] + sq_norms[None,:]
                sq_median_distance = da.percentile(sq_distances.ravel(), 50)
                return da.exp((-sq_distances / (2*threshold**2 * sq_median_distance)))
            else:
                raise ValueError

        # Set `limit`
        n_words = next(iter(self.representations_d.values())).size()[0]
        if type(self.limit) == float:
            limit = int(n_words * self.limit)
        elif type(self.limit) == int:
            limit = self.limit
        else:
            limit = self.limit

        # Set `daskp`
        # Logic could become more complex
        daskp = True if self.device == torch.device('cpu') else False

        # dask setup
        if daskp:
            from dask.distributed import Client, progress
            client = Client(processes=False, threads_per_worker=32, n_workers=1, memory_limit='200GB')

        # Set `self.similarities`
        # {network: {other: rbfcka_similarity}}
        self.similarities = {network: {} for network in self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                           desc='rbfcka',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            if other_network in self.similarities[network]: 
                continue

            if daskp:
                c = self.dask_chunk_size
                X = da.from_array(np.asarray(self.representations_d[network][:limit]), chunks=(c, c))
                Y = da.from_array(np.asarray(self.representations_d[other_network][:limit]), chunks=(c, c))

                Gx = center_gram(gram_rbf(X))
                Gy = center_gram(gram_rbf(Y))

                scaled_hsic = da.dot(Gx.ravel(), Gy.ravel())
                norm_gx = da.sqrt(da.dot(Gx.ravel(), Gx.ravel()))
                norm_gy = da.sqrt(da.dot(Gy.ravel(), Gy.ravel()))

                sim = (scaled_hsic / (norm_gx*norm_gy)).compute()
            else:
                device = self.device
                X = self.representations_d[network][:limit].to(device)
                Y = self.representations_d[other_network][:limit].to(device)

                # TO DO: random subset of data using limit?
                Gx = center_gram(gram_rbf(X))
                Gy = center_gram(gram_rbf(Y))

                scaled_hsic = torch.dot(Gx.view(-1), Gy.view(-1)).cpu().item()
                norm_gx = torch.norm(Gx, p="fro").cpu().item()
                norm_gy = torch.norm(Gy, p="fro").cpu().item()

                sim = scaled_hsic / (norm_gx*norm_gy)

            self.similarities[network][other_network] = sim
            self.similarities[other_network][network] = sim




    def compute_correlations(self):
        """
        Set `self.similarities`. 
        """
        # Center
        if self.normalize_dimensions:
            for network in tqdm(self.representations_d, desc='mu, sigma'):
                t = self.representations_d[network]
                means = t.mean(0, keepdim=True)

                self.representations_d[network] = t - means

        # Set `self.similarities`
        # {network: {other: lincka_similarity}}
        self.similarities = {network: {} for network in
                             self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                           desc='lincka',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            if other_network in self.similarities[network]: 
                continue

            X = self.representations_d[network].to(self.device)
            Y = self.representations_d[other_network].to(self.device)

            XtX_F = torch.norm(torch.mm(X.t(), X), p='fro').item()
            YtY_F = torch.norm(torch.mm(Y.t(), Y), p='fro').item()
            YtX_F = torch.norm(torch.mm(Y.t(), X), p='fro').item()

            # eq 5 in paper
            sim = YtX_F**2 / (XtX_F*YtY_F)
            self.similarities[network][other_network] = sim
            self.similarities[other_network][network] = sim

    def write_correlations(self, output_file):
        torch.save(self.similarities, output_file)

    def __str__(self):
        return "lincka"
        

class RBFCKA(Method):
    def __init__(self, num_neurons_d, representations_d, device=None,
                 limit=None, dask_chunk_size=50_000):
        super().__init__(num_neurons_d, representations_d, device)
        self.limit = limit
        self.dask_chunk_size = dask_chunk_size

    def compute_correlations(self):
        def center_gram(G):
            means = G.mean(0)
            means -= means.mean() / 2
            return G - means[None, :] - means[:, None]

        def gram_rbf(X, threshold=1.0):
            if type(X) == torch.Tensor:
                dot_products = X @ X.t()
                sq_norms = dot_products.diag()
                sq_distances = -2*dot_products + sq_norms[:,None] + sq_norms[None,:]
                sq_median_distance = sq_distances.median()
                return torch.exp(-sq_distances / (2*threshold**2 * sq_median_distance))
            elif type(X) == da.Array:
                dot_products = X @ X.T
                sq_norms = da.diag(dot_products)
                sq_distances = -2*dot_products + sq_norms[:,None] + sq_norms[None,:]
                sq_median_distance = da.percentile(sq_distances.ravel(), 50)
                return da.exp((-sq_distances / (2*threshold**2 * sq_median_distance)))
            else:
                raise ValueError

        # Set `limit`
        n_words = next(iter(self.representations_d.values())).size()[0]
        if type(self.limit) == float:
            limit = int(n_words * self.limit)
        elif type(self.limit) == int:
            limit = self.limit
        else:
            limit = self.limit

        # Set `daskp`
        # Logic could become more complex
        daskp = True if self.device == torch.device('cpu') else False

        # dask setup
        if daskp:
            from dask.distributed import Client, progress
            client = Client(processes=False, threads_per_worker=32,
                            n_workers=1, memory_limit='160GB',
                            local_dir="/data/sls/temp/johnmwu/dask-worker-space")

        # Set `self.similarities`
        # {network: {other: rbfcka_similarity}}
        self.similarities = {network: {} for network in self.representations_d}
        for network, other_network in tqdm(p(self.representations_d,
                                             self.representations_d),
                                           desc='rbfcka',
                                           total=len(self.representations_d)**2):

            if network == other_network:
                continue

            if other_network in self.similarities[network]: 
                continue

            if daskp:
                c = self.dask_chunk_size
                X = da.from_array(np.asarray(self.representations_d[network][:limit]), chunks=(c, c))
                Y = da.from_array(np.asarray(self.representations_d[other_network][:limit]), chunks=(c, c))

                Gx = center_gram(gram_rbf(X))
                Gy = center_gram(gram_rbf(Y))

                scaled_hsic = da.dot(Gx.ravel(), Gy.ravel())
                norm_gx = da.sqrt(da.dot(Gx.ravel(), Gx.ravel()))
                norm_gy = da.sqrt(da.dot(Gy.ravel(), Gy.ravel()))

                sim = (scaled_hsic / (norm_gx*norm_gy)).compute()
            else:
                device = self.device
                X = self.representations_d[network][:limit].to(device)
                Y = self.representations_d[other_network][:limit].to(device)

                # TO DO: random subset of data using limit?
                Gx = center_gram(gram_rbf(X))
                Gy = center_gram(gram_rbf(Y))

                scaled_hsic = torch.dot(Gx.view(-1), Gy.view(-1)).cpu().item()
                norm_gx = torch.norm(Gx, p="fro").cpu().item()
                norm_gy = torch.norm(Gy, p="fro").cpu().item()

                sim = scaled_hsic / (norm_gx*norm_gy)

            self.similarities[network][other_network] = sim
            self.similarities[other_network][network] = sim


    def write_correlations(self, output_file):
        torch.save(self.similarities, output_file)

    def __str__(self):
        return "rbfcka"
        
