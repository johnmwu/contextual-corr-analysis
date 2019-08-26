import torch 
from tqdm import tqdm
from itertools import product as p
import json
import numpy as np
import h5py
from os.path import basename, dirname
import dask.array as da
import pickle
from var import fname2mname

def load_attentions(attention_fname_l, limit=None,
                         layerspec_l=None):
    """
    Load in attentions. Options to control loading exist. 

    Params:
    ----
    attention_fname_l : list<str>
        List of hdf5 files containing attentions
    limit : int or None
        Limit on number of attentions to take
    layerspec_l : list
        Specification for each model. May be an integer (layer to take),
        or "all" or "full". "all" means take all layers. "full" means to
        concatenate all layers together.

    Returns:
    ----
    num_head_d : {str : int}
        {network : number of heads}. Here a network could be a layer,
        or the stack of all layers, etc. A network is what's being
        correlated as a single unit.
    attentions_d : {str : list<tensor>}
        {network : attentions}. attentions is a list because each 
        sentence may be of different length. 
    """

    # Edit args
    l = len(attention_fname_l)
    if layerspec_l is None:
        layerspec_l = ['all'] * l

    # Main loop
    num_heads_d = {} 
    attentions_d = {} 
    for loop_var in tqdm(zip(attention_fname_l, layerspec_l)):
        fname, layerspec = loop_var

        # Set `attentions_h5`, `sentence_d`, `indices`
        attentions_h5 = h5py.File(fname, 'r')
        sentence_d = json.loads(attentions_h5['sentence_to_index'][0])
        temp = {} # TO DO: Make this more elegant?
        for k, v in sentence_d.items():
            temp[v] = k
        sentence_d = temp # {str ix, sentence}
        indices = list(sentence_d.keys())[:limit]

        # Set `num_layers`, `num_heads`, `layers`
        s = attentions_h5[indices[0]].shape
        num_layers = s[0]
        num_heads = s[1]
        if layerspec == "all":
            layers = list(range(num_layers))
        elif layerspec == "full":
            layers = ["full"]
        else:
            layers = [layerspec]

        # Set `num_heads_d`, `attentions_d`
        for layer in layers:
            # Create `attentions_l`
            attentions_l = []
            word_count = 0
            for sentence_ix in indices: 
                # Set `dim`, `n_word`, update `word_count`
                shape = attentions_h5[sentence_ix].shape
                dim = len(shape)
                if not (dim == 4):
                    raise ValueError('Improper array dimension in file: ' +
                                     fname + "\nShape: " +
                                     str(attentions_h5[sentence_ix].shape))
                n_word = shape[2]
                word_count += n_word

                # Create `attentions`
                if layer == "full":
                    attentions = torch.FloatTensor(
                        attentions_h5[sentence_ix])
                    attentions = attentions.contiguous().view(
                    	-1, n_word, n_word)
                else:
                    attentions = torch.FloatTensor(
                        attentions_h5[sentence_ix][layer]
                    )

                # Update `attentions_l`
                attentions_l.append(attentions)

                # Early stop
                if limit is not None and word_count >= limit:
                    break

            # Main update
            network = "{mname}_{layer}".format(mname=fname2mname(fname), 
                                                  layer=layer)
            num_heads_d[network] = attentions_l[0].shape[0]
            attentions_d[network] = attentions_l[:limit] 
    
    return num_heads_d, attentions_d


class Method(object):
    """
    Abstract representation of a correlation method. 

    Example instances are MaxCorr, MinCorr, MaxLinReg, MinLinReg, CCA,
    LinCKA.
    """
    def __init__(self, num_heads_d, attentions_d, device=None):
        self.num_heads_d = num_heads_d
        self.attentions_d = attentions_d
        self.device = device

    def compute_correlations(self):
        raise NotImplementedError

    def write_correlations(self):
        raise NotImplementedError


class MaxMinCorr(Method):
    def __init__(self, num_neurons_d, representations_d, device,
                 op=None):
        self.op = op

    def compute_correlations(self):
        # Set `self.corrs` : {network: {other: [corr]}}
        # Set `self.pairs` : {network: {other: [pair]}}
        # pair is index of head in other network
        # Set `self.similarities` : {network: {other: sim}}
        self.corrs = {network: {} for network in
                             self.attentions_d}
        self.pairs = {network: {} for network in
                             self.attentions_d}
        self.similarities = {network: {} for network in
                         self.attentions_d}
        # TODO: remove 
        num_sentences = len(next(iter(self.attentions_d.values())))
        for network, other_network in tqdm(p(self.attentions_d,
                                             self.attentions_d),
                                             desc='correlate',
                                             total=len(self.attentions_d)**2):
            if network == other_network:
                continue

            if other_network in self.corrs[network]: 
                continue

            device = self.device

            # TODO: make this more efficient? 
            num_heads_network = self.attentions_d[network][0].shape[0]
            num_heads_other_network = self.attentions_d[other_network][0].shape[0]
            correlation = np.ones((num_sentences, num_heads_network, num_heads_other_network))

            # loop over sentences (may have different lengths)
            for idx, (network_attentions, other_network_attentions) in enumerate(zip(self.attentions_d[network], self.attentions_d[other_network])):
                for network_head, other_network_head in p(range(len(network_attentions)), range(len(other_network_attentions))):

                    if network_head == other_network_head:
                        continue

                    t1 = network_attentions[network_head].to(device)
                    t2 = other_network_attentions[other_network_head].to(device)
                    # TODO: consider other norms (perhaps also asymmetric measures?)
                    distance = torch.norm(t1 - t1, p='fro')
                    distance = distance.cpu().numpy().item()
                    similarity = 1 - distance
                    correlation[i][network_head][other_network_head] = similarity

            self.corrs[network][other_network] = correlation.max(axis=1)
            self.corrs[other_network][network] = correlation.max(axis=0)

            self.similarities[network][other_network] = self.corrs[network][other_network].mean()
            self.similarities[other_network][network] = self.corrs[other_network][network].mean()

            self.pairs[network][other_network] = correlation.argmax(axis=1)
            self.pairs[other_network][network] = correlation.argmax(axis=0)

        # Set `self.head_sort` : {network, sorted_list}
        # Set `self.head_notated_sort` : {network: [(head, {other: (corr, pair)})]}
        self.head_sort = {} 
        self.head_notated_sort = {}
        for network in tqdm(self.attentions_d, desc='annotation'):
            self.head_sort[network] = sorted(
                range(self.num_heads_d[network]), 
                key=lambda i: self.op(
                    self.corrs[network][other][i] for other in self.corrs[network]
                ), 
                reverse=True,
            )
            self.head_notated_sort[network] = [
                (
                    head,
                    {
                        other : (
                            self.corrs[network][other][head], 
                            self.pairs[network][other][head],
                        ) 
                        for other in self.corrs[network]
                    }
                ) 
                for head in self.head_sort[network]
            ]

    def write_correlations(self, output_file):
        output = {
            "corrs" : self.corrs, 
            "pairs" : self.pairs,
            "similarities" : self.similarities,
            "head_sort" : self.head_sort, 
            "head_notated_sort" : self.head_notated_sort,
        }

        with open(output_file, "wb") as f:
            pickle.dump(output, f)


class MaxCorr(MaxMinCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device, op=max)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "maxcorr"


class MinCorr(MaxMinCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device, op=min)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "mincorr"




# TODO: probably remove all this
class LinReg(Method): 
    def __init__(self, num_neurons_d, representations_d, device=None, op=None):
        super().__init__(num_neurons_d, representations_d, device)
        self.op = op

    def compute_correlations(self):
        # Set `means_d`, `stdevs_d`
        # Set `self.nrepresentations_d` to be normalized. 
        means_d = {}
        stdevs_d = {}
        self.nrepresentations_d = {}
        self.lsingularv_d = {}

        for network in tqdm(self.representations_d, desc='mu, sigma'):
            t = self.representations_d[network].to(self.device)
            means = t.mean(0, keepdim=True)
            stdevs = (t - means).pow(2).mean(0, keepdim=True).pow(0.5)

            means_d[network] = means.cpu()
            stdevs_d[network] = stdevs.cpu()
            self.nrepresentations_d[network] = ((t - means) / stdevs).cpu()
            self.lsingularv_d[network], _, _ = torch.svd(self.nrepresentations_d[network])

            self.representations_d[network] = None # free up memory

        # Set `self.pred_power`
        # If the data is centered, it is the r value.
        # Set `self.similarities`
        self.pred_power = {network: {} for network in self.nrepresentations_d}
        self.similarities = {network: {} for network in self.nrepresentations_d}        
        for network, other_network in tqdm(p(self.nrepresentations_d,
                                             self.nrepresentations_d),
                                           desc='correlate',
                                           total=len(self.nrepresentations_d)**2):

            if network == other_network:
                continue

            U = self.lsingularv_d[other_network].to(self.device)
            Y = self.nrepresentations_d[network].to(self.device)

            # SVD method of linreg
            UtY = torch.mm(U.t(), Y) # b for Ub = Y

            bnorms = torch.norm(UtY, dim=0)
            ynorms = torch.norm(Y, dim=0)

            self.pred_power[network][other_network] = (bnorms / ynorms).cpu().numpy()
            self.similarities[network][other_network] = self.pred_power[network][other_network].mean()


        # Set `self.neuron_sort` : {network: sorted_list}
        # Set `self.neuron_notated_sort` : {network: [(neuron, {other_network: pred_power})]}
        self.neuron_sort = {}
        self.neuron_notated_sort = {}
        # Sort neurons by correlation with another network
        for network in tqdm(self.nrepresentations_d, desc='annotation'):
            self.neuron_sort[network] = sorted(
                    range(self.num_neurons_d[network]),
                    key = lambda i: self.op(
                        self.pred_power[network][other][i] 
                        for other in self.pred_power[network]),
                    reverse=True
                )

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

    def write_correlations(self, output_file):
        output = {
            "pred_power" : self.pred_power,
            "similarities" : self.similarities,
            "neuron_sort" : self.neuron_sort,
            "neuron_notated_sort" : self.neuron_notated_sort,    
        }

        with open(output_file, "wb") as f:
            pickle.dump(output, f)

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
    """
    Compute SVCCA and PWCCA. 

    See the papers 
    - SVCCA: https://arxiv.org/abs/1706.05806
    - PWCCA: https://arxiv.org/abs/1806.05759
    """

    def __init__(self, num_neurons_d, representations_d, device=None,
                 percent_variance=0.99, normalize_dimensions=True,
                 save_cca_transforms=False):
        super().__init__(num_neurons_d, representations_d, device)

        self.percent_variance = percent_variance
        self.normalize_dimensions = normalize_dimensions
        self.save_cca_transforms = save_cca_transforms

    def compute_correlations(self):
        # Set `self.nrepresentations_d`, "normalized representations".
        # Call it this regardless of if it's actually centered or scaled
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
            X = self.nrepresentations_d[network]
            U, S, V = torch.svd(X)

            var_sums = torch.cumsum(S.pow(2), 0)
            wanted_size = torch.sum(var_sums.lt(var_sums[-1] *
                                                self.percent_variance)).item()

            print('For network', network, 'wanted size is', wanted_size)

            if self.save_cca_transforms:
                whitening_transform = torch.mm(V, torch.diag(1/S))
                whitening_transforms[network] = whitening_transform[:, :wanted_size]

            pca_directions[network] = U[:, :wanted_size]

        # Set 
        # `self.transforms`: {network: {other: svcca_transform}}
        # `self.corrs`: {network: {other: canonical_corrs}}
        # `self.pw_alignments`: {network: {other: unnormalized pw weights}}
        # `self.pw_corrs`: {network: {other: pw_alignments*corrs}}
        # `self.sv_similarities`: {network: {other: svcca_similarities}}
        # `self.pw_similarities`: {network: {other: pwcca_similarities}}
        self.transforms = {network: {} for network in self.nrepresentations_d}
        self.corrs = {network: {} for network in self.nrepresentations_d}
        self.pw_alignments = {network: {} for network in
                              self.nrepresentations_d}
        self.pw_corrs = {network: {} for network in self.nrepresentations_d}
        self.sv_similarities = {network: {} for network in
                                self.nrepresentations_d}
        self.pw_similarities = {network: {} for network in
                                self.nrepresentations_d}
        for network, other_network in tqdm(p(self.nrepresentations_d,
                                             self.nrepresentations_d),
                                           desc='cca',
                                           total=len(self.nrepresentations_d)**2):

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

            # `self.transforms`, `self.corrs`, `self.sv_similarities`
            if self.save_cca_transforms:
                self.transforms[network][other_network] = torch.mm(whitening_transforms[network], u).cpu().numpy()
                self.transforms[other_network][network] = torch.mm(whitening_transforms[other_network], v).cpu().numpy()

            self.corrs[network][other_network] = s.cpu().numpy()
            self.corrs[other_network][network] = s.cpu().numpy()

            self.sv_similarities[network][other_network] = s.mean().item()
            self.sv_similarities[other_network][network] = s.mean().item()

            # Compute `self.pw_alignments`, `self.pw_corrs`, `self.pw_similarities`. 
            # This is not symmetric

            # For X
            H = torch.mm(X, u)
            Z = self.representations_d[network]
            align = torch.abs(torch.mm(H.t(), Z))
            a = torch.sum(align, dim=1, keepdim=False)
            self.pw_alignments[network][other_network] = a.cpu().numpy()
            self.pw_corrs[network][other_network] = (s*a).cpu().numpy()
            self.pw_similarities[network][other_network] = (torch.sum(s*a)/torch.sum(a)).item()

            # For Y
            H = torch.mm(Y, v)
            Z = self.representations_d[other_network]
            align = torch.abs(torch.mm(H.t(), Z))
            a = torch.sum(align, dim=1, keepdim=False)
            self.pw_alignments[other_network][network] = a.cpu().numpy()
            self.pw_corrs[other_network][network] = (s*a).cpu().numpy()
            self.pw_similarities[other_network][network] = (torch.sum(s*a)/torch.sum(a)).item()

    def write_correlations(self, output_file):
        if self.save_cca_transforms:
            output = {
                "transforms": self.transforms,
                "corrs": self.corrs,
                "sv_similarities": self.sv_similarities,
                "pw_alignments": self.pw_alignments,
                "pw_corrs": self.pw_corrs,
                "pw_similarities": self.pw_similarities,
            }
        else:
            output = {
                "corrs": self.corrs,
                "sv_similarities": self.sv_similarities,
                "pw_alignments": self.pw_alignments,
                "pw_corrs": self.pw_corrs,
                "pw_similarities": self.pw_similarities,
            }
        with open(output_file, "wb") as f:
            pickle.dump(output, f)

    def __str__(self):
        return "cca"

class LinCKA(Method):
    """
    See the paper: https://arxiv.org/abs/1905.00414. 

    This and RBFCKA don't inherit from the same method because there's a
    trick for LinCKA which speeds up computation. 
    """

    def __init__(self, num_neurons_d, representations_d, device=None,
                 normalize_dimensions=True):
        super().__init__(num_neurons_d, representations_d, device)
        self.normalize_dimensions = normalize_dimensions

    def compute_correlations(self):
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
        output = {
            "similarities": self.similarities,
        }

        with open(output_file, "wb") as f:
            pickle.dump(output, f)

    def __str__(self):
        return "lincka"
        

class RBFCKA(Method):
    """
    See the paper: https://arxiv.org/abs/1905.00414
    """

    def __init__(self, num_neurons_d, representations_d, device=None,
                 dask_chunk_size=25_000):
        super().__init__(num_neurons_d, representations_d, device)
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

        # Set `daskp`
        daskp = True if self.device == torch.device('cpu') else False
        print("daskp value: {0}".format(daskp))

        # Set `self.similarities`
        # {network: {other: rbfcka_similarity}}
        self.similarities = {network: {} for network in
                             self.representations_d}
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
                X = da.from_array(np.asarray(self.representations_d[network]), chunks=(c, c))
                Y = da.from_array(np.asarray(self.representations_d[other_network]), chunks=(c, c))

                Gx = center_gram(gram_rbf(X))
                Gy = center_gram(gram_rbf(Y))

                scaled_hsic = da.dot(Gx.ravel(), Gy.ravel())
                norm_gx = da.sqrt(da.dot(Gx.ravel(), Gx.ravel()))
                norm_gy = da.sqrt(da.dot(Gy.ravel(), Gy.ravel()))

                sim = (scaled_hsic / (norm_gx*norm_gy)).compute()
            else:
                device = self.device
                X = self.representations_d[network].to(device)
                Y = self.representations_d[other_network].to(device)

                Gx = center_gram(gram_rbf(X))
                Gy = center_gram(gram_rbf(Y))

                scaled_hsic = torch.dot(Gx.view(-1), Gy.view(-1)).cpu().item()
                norm_gx = torch.norm(Gx, p="fro").cpu().item()
                norm_gy = torch.norm(Gy, p="fro").cpu().item()

                sim = scaled_hsic / (norm_gx*norm_gy)

            self.similarities[network][other_network] = sim
            self.similarities[other_network][network] = sim


    def write_correlations(self, output_file):
        output = {
            "similarities": self.similarities,
        }

        with open(output_file, "wb") as f:
            pickle.dump(output, f)

    def __str__(self):
        return "rbfcka"
