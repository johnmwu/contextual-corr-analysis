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
