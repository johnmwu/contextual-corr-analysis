#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#from sklearn.datasets import load_iris, load_digits
#from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

import umap

import sys
sys.path.append('..')
from corr_methods import load_representations
from MCCA import MCCA
import pickle
import time


start_time = time.time()

# # Load data

# In[2]:

print('load data', flush=True)
representations_filename_l = [
"/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/bert_large_cased/ptb_pos_dev.hdf5", 
"/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/openai_transformer/ptb_pos_dev.hdf5",
"/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/bert_base_cased/ptb_pos_dev.hdf5",
"/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/elmo_original/ptb_pos_dev.hdf5",
"/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/calypso_transformer_6_512_base/ptb_pos_dev.hdf5",
"/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/elmo_4x4096_512/ptb_pos_dev.hdf5",
"/data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/xlnet_large_cased/ptb_pos_dev.hdf5"
]
layerspec_l = ["all" for x in representations_filename_l]
first_half_only_l = [False for x in representations_filename_l]
second_half_only_l = [False for x in representations_filename_l]
a = load_representations(representations_filename_l, limit=10, layerspec_l=layerspec_l, 
                         first_half_only_l=first_half_only_l, second_half_only_l=second_half_only_l)
num_neurons_d, representations_d = a
# for name in representations_d:
#     print(name, representations_d[name].shape)
# print(representations_d.keys())
# print(list(representations_d.keys())[25])
#print(representations_d['openai_transformer-ptb_pos_dev.hdf5_6'].numpy().flatten().shape)
representations_a = [representations_d[name] for name in representations_d]
# print(type(representations_a[0]), flush=True)
# print(representations_a[0].numpy(), flush=True)
representations_a = [representations.numpy() for representations in representations_a]
print(representations_d.keys())
print(len(representations_a), flush=True)
print(representations_a[0], flush=True)
print(representations_a[0].shape)
# print(type(representations_a[0]))
# representations_a = np.array(representations_a, dtype='float32')
# print(type(representations_a))
# representations_a


# # Run MCCA

# In[3]:

print('run mcca', flush=True)
# representations_a = representations_a[59:61]
# representations_a = representations_a[:40]
to_keep = [0, 1, 11, 12, 23, 24, # bert large
        24+1, 24+2, 24+6, 24+7, 24+12, 24+13, # gpt1
        37+1, 37+2, 37+6, 37+7, 37+12, 37+13, # bert base
        50+1, 50+2, 50+3, # elmo original
        53+1, 53+2, 53+5, 53+7, # calypso
        60+1, 60+2, 60+5, # elmo 4x4096
        65+1, 65+2, 65+12, 65+13, 65+24, 65+25] # xlnet large
print(to_keep)

#representations_a = [representations_a[i] for i in range(len(representations_a)) if i % 3 == 0]
representations_a = [representations_a[i] for i in range(len(representations_a)) if i in to_keep]
representations_a = [r[:, :30] for r in representations_a]
print(len(representations_a))
print(representations_a[0].shape)
print(representations_a[0][:3,:3])
num_views = len(representations_a)
print(num_views)
proj_dim = 1
rs = [0.00001]*num_views
print(np.any([np.any(np.isnan(r)) for r in representations_a]))
print(np.any([np.any(np.isinf(r)) for r in representations_a]))
# print('hi')
# for r in representations_a:
#     print(np.any(np.isinf(r)))
print(representations_a[0].shape)
# for i in range(100):
#     for j in range(100):
#         print(representations_a[0][i][j], end=' ')
#     print('')
# sns.heatmap(data=representations_a[0])


# In[4]:


# with open('./mcca_args.pkl', 'wb') as f:
#     data_tuple = (num_views, proj_dim, representations_a, rs)
#     pickle.dump(data_tuple, f, -1)


# In[5]:


CORR, UU, MM, X_PROJ = MCCA(num_views, proj_dim, representations_a, rs)


# In[6]:


print(len(CORR)); print(CORR)
print(len(UU)); print(UU)
print(len(MM)); print(MM)
print(len(X_PROJ)); print(X_PROJ)
print(X_PROJ[0].shape); # print(X_PROJ[1].shape); print(X_PROJ[2].shape)


# # Run UMAP

# In[7]:

print('run umap', flush=True)
X_PROJ_flat = np.array([x.real.flatten() for x in X_PROJ])
print(X_PROJ_flat)
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_PROJ_flat)
embedding.shape


# # Plot

# In[8]:

print('plot', flush=True)
model_names = [name.split('-')[0] for name in representations_d] #[:40]
layers = [int(name.split('_')[-1]) for name in representations_d] #[:40]
#model_names = [model_names[i] for i in range(len(model_names)) if i % 3 == 0]
#layers = [layers[i] for i in range(len(layers)) if i % 3 == 0]
model_names = [model_names[i] for i in range(len(model_names)) if i in to_keep]
layers = [layers[i] for i in range(len(layers)) if i in to_keep]
print(model_names)
print(layers)
model_names_unique = list(set(model_names))
name_idx_d = dict(zip(model_names_unique, range(len(model_names_unique))))
model_idx = [name_idx_d[name] for name in model_names]
print(name_idx_d)
print(model_idx)
print(model_names)
# print(model_names[59])


df = pd.DataFrame({'layer': layers, 'model': model_names, 
                   'x': embedding[:, 0], 'y': embedding[:, 1]})
print(df)

scatter = sns.scatterplot(x='x', y='y', hue='model', data=df)
for i, layer in enumerate(layers):
    scatter.annotate(layer, (embedding[i, 0], embedding[i, 1]))
#scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in model_idx])
plt.legend(framealpha=0.5)
plt.gca().set_aspect('equal', 'datalim')
#plt.title('UMAP projection of activations', fontsize=24);
plt.savefig('mcca.png') 


# In[ ]:
print('done')
elapsed_time = time.time() - start_time
print('total time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


