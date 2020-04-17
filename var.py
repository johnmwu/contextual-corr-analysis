from os.path import basename, dirname

def fname2mname(fname):
    """
    Filename to model name. 

    The nature of this function depends on where your data is located. 

    For contextual-corr-analysis, an example is
        /data/sls/temp/belinkov/contextual-corr-analysis/contextualizers/bert_large_cased/ptb_pos_dev.hdf5
        -> 
        bert_large_cased-ptb_pos_dev.hdf5
 
    We take the filename too, because some models have multiple runs in them. 
    """
    return '-'.join([basename(dirname(fname)), basename(fname)])

# The following functions may be reimplemented if desired, particularly if
# you've changed the formatting
def network2pair(network):
    """
    Takes in a network name, and returns (mname, layer) pair. 

    `layer` could be an int or one of the strings "full", .
    """
    i = network.rfind('_')
    mname = network[:i]
    layer = network[i+1:]
    try:
        layer = int(layer)
    except:
        pass

    return mname, layer

def network_sort_key(network):
    """
    So there's a consistent global ordering of our networks. 
    """
    mname, layer = network2pair(network)
    if layer == "full":
        layer = 0

    return mname, layer
