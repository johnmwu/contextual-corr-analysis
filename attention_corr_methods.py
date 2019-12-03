import torch 
from tqdm import tqdm
from itertools import product as p
import json
import numpy as np
import h5py
from os.path import basename, dirname
#import dask.array as da
import pickle
from var import fname2mname

def load_attentions(attention_fname_l, limit=None, layerspec_l=None,
                    ar_mask=False):
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
        or "all". "all" means take all layers. 
    ar_mask : bool
        Whether to mask the future when loading. Some models (eg. gpt)
        do this automatically.

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
    for loop_var in tqdm(zip(attention_fname_l, layerspec_l), desc='load'):
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
                if ar_mask:
                    attentions = np.tril(attentions_h5[sentence_ix][layer])
                    attentions = attentions/np.sum(attentions, axis=-1,
                                                   keepdims=True)
                    attentions = torch.FloatTensor(attentions)
                else:
                    attentions = torch.FloatTensor(
                        attentions_h5[sentence_ix][layer] )

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
    def __init__(self, num_heads_d, attentions_d, device, op=None):
        super().__init__(num_heads_d, attentions_d, device)
        self.op = op

    def correlation_matrix(self, network, other_network):
        raise NotImplementedError

    def compute_correlations(self):
        # convenient variables
        device = self.device
        self.num_sentences = len(next(iter(self.attentions_d.values())))
        self.num_words = sum(t.size()[-1] for t in
                             next(iter(self.attentions_d.values())))

        # Set `self.corrs` : {network: {other: [corr]}}
        # Set `self.pairs` : {network: {other: [pair]}}
        # pair is index of head in other network
        # Set `self.similarities` : {network: {other: sim}}
        self.corrs = {network: {} for network in self.attentions_d}
        self.pairs = {network: {} for network in self.attentions_d}
        self.similarities = {network: {} for network in self.attentions_d}
        for network, other_network in tqdm(p(self.attentions_d,
                                             self.attentions_d),
                                             desc='correlate',
                                             total=len(self.attentions_d)**2):
            if network == other_network:
                continue

            if other_network in self.corrs[network]: 
                continue

            correlation = self.correlation_matrix(network, other_network)

            # Main update
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


class FroMaxMinCorr(MaxMinCorr):
    """
    A MaxMinCorr method based on taking Frobenius norms.
    """
    def correlation_matrix(self, network, other_network):
        device = self.device
        num_sentences = self.num_sentences

        distances = np.zeros((num_sentences, self.num_heads_d[network],
                              self.num_heads_d[other_network]))
        for idx, (attns, o_attns) in enumerate(
                zip(self.attentions_d[network],
                    self.attentions_d[other_network])):
            t1 = attns.to(device)
            t2 = o_attns.to(device)
            t11, t12, t13 = t1.size()
            t21, t22, t23 = t2.size()
            t1 = t1.reshape(t11, 1, t12, t13)
            t2 = t2.reshape(1, t21, t22, t23)

            distance = torch.norm(t1-t2, p='fro', dim=(2,3))
            distances[idx] = distance.cpu().numpy()

        # Set `correlation`
        distances = distances.mean(axis=0)
        correlation = 1 - distances

        return correlation


class MaxCorr(FroMaxMinCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device, op=max)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "maxcorr"


class MinCorr(FroMaxMinCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device, op=min)

    def compute_correlations(self):
        super().compute_correlations()

    def __str__(self):
        return "mincorr"


class PearsonMaxMinCorr(MaxMinCorr):
    """
    A MaxMinCorr method based on taking Pearson correlations. 
    """
    def correlation_matrix(self, network, other_network):
        device = self.device
        num_sentences = self.num_sentences

        # set `total_corrs`
        total_corrs = np.zeros((num_sentences, self.num_heads_d[network],
                                self.num_heads_d[other_network]))
        for idx, (attns, o_attns) in enumerate(
                zip(self.attentions_d[network],
                    self.attentions_d[other_network])):
            t1 = attns.to(device)
            t2 = o_attns.to(device)
            t11, t12, t13 = t1.size()
            t21, t22, t23 = t2.size()
            t1 = t1.reshape(t11, 1, t12, t13)
            t2 = t2.reshape(1, t21, t22, t23)

            if t12 < 2: # t12 = sentence length
                continue

            cov = (t1*t2).mean(dim=-1) - t1.mean(dim=-1)*t2.mean(dim=-1)
            corr = cov/(torch.std(t1, dim=-1)*torch.std(t2, dim=-1))
            total_corrs[idx] = corr.sum(dim=-1).cpu().numpy()

        # set `correlation`
        correlation = total_corrs.sum(axis=0)/self.num_words
        return correlation


class PearsonMaxCorr(PearsonMaxMinCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device, op=max)

    def __str__(self):
        return "pearsonmaxcorr"


class PearsonMinCorr(PearsonMaxMinCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device, op=min)

    def __str__(self):
        return "pearsonmincorr"


class JSMaxMinCorr(MaxMinCorr):
    """
    A MaxMinCorr method based on Jensen-Shannon divergence. 
    """
    def correlation_matrix(self, network, other_network):
        device = self.device
        num_sentences = self.num_sentences

        # set `total_corrs`
        total_dist = np.zeros((num_sentences, self.num_heads_d[network],
                                      self.num_heads_d[other_network]))
        for idx, (attns, o_attns) in enumerate(
                zip(self.attentions_d[network],
                    self.attentions_d[other_network])):
            t1 = attns.to(device)
            t2 = o_attns.to(device)
            t11, t12, t13 = t1.size()
            t21, t22, t23 = t2.size()
            t1 = t1.reshape(t11, 1, t12, t13)
            t2 = t2.reshape(1, t21, t22, t23)

            # set `kl1`, `kl2`
            m = (t1+t2)/2
            kl1s = t1*(torch.log(t1) - torch.log(m))
            kl1s[torch.isnan(kl1s)] = 0 
            kl1 = torch.sum(kl1s, dim=-1) 
            kl2s = t2*(torch.log(t2) - torch.log(m))
            kl2s[torch.isnan(kl2s)] = 0 
            kl2 = torch.sum(kl2s, dim=-1) 

            js = (kl1 + kl2)/2 
            total_dist[idx] = js.sum(dim=-1).cpu().numpy()

        # set `correlation`
        correlation = 1 - total_dist.sum(axis=0)/self.num_words
        return correlation


class JSMaxCorr(JSMaxMinCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device, op=max)

    def __str__(self):
        return "jsmaxcorr"

class JSMinCorr(JSMaxMinCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device, op=min)

    def __str__(self):
        return "jsmincorr"


class ReprCorr(Method):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device)

        # set `self.representations_d`
        self.representations_d = {}
        for network, al in self.attentions_d.items():
            fal = [torch.flatten(at, start_dim=1).t() for at in al]
            self.representations_d[network] = torch.cat(fal)

    
class AttnLinCKA(ReprCorr):
    def __init__(self, num_heads_d, attentions_d, device):
        super().__init__(num_heads_d, attentions_d, device)
    
    def compute_correlations(self):
        # Copied and pasted from `corr_methods.py`. Ideally, we'd share
        # the code. 

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
        return "attn_lincka"


class AttnCCA(ReprCorr):
    def __init__(self, num_heads_d, attentions_d, device,
                 percent_variance=0.99, normalize_dimensions=True,
                 save_cca_transforms=False):
        super().__init__(num_heads_d, attentions_d, device)

        self.percent_variance = percent_variance
        self.normalize_dimensions = normalize_dimensions
        self.save_cca_transforms = save_cca_transforms
    
    def compute_correlations(self):
        # Copied from `corr_methods.py`. 
        
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
        # Copied from `corr_methods.py`. 

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
        return "attn_cca"
            
