# inspired from https://github.com/kexinhuang12345/MolTrans

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from subword_nmt.apply_bpe import BPE
import codecs

# build query dict
p_file = codecs.open('./ESPF/protein_codes_uniprot.txt')
p_bpe = BPE(p_file, merges=-1, separator='')
p_subword = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')
p_idxtoword = p_subword['index'].values
p_wordtoidx = dict(zip(p_idxtoword, range(0, len(p_idxtoword))))

d_file = codecs.open('./ESPF/drug_codes_chembl.txt')
d_bpe = BPE(d_file, merges=-1, separator='')
d_subword = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
d_idxtoword = d_subword['index'].values
d_wordtoidx = dict(zip(d_idxtoword, range(0, len(d_idxtoword))))

#max_d = 205
#max_p = 545


def drug_encoder(d):
    max_d = 50
    d_onebpe = d_bpe.process_line(d).split()
    try:
        v = np.asarray([d_wordtoidx[i] for i in d_onebpe])
        #print(v.shape)
    except:
        v = np.array([0])
        print(d)

    # mask
    length = len(v)
    #print(length)
    if length<max_d:
        v2 = np.pad(v,(0,max_d-length),'constant',constant_values=0)
        #print(v2.shape)
        mask = ([1]*length)+([0]*(max_d-length))
    else:
        # cut to maxlength
        v2 = v[:max_d]
        #print(v2.shape)
        mask = [1]*max_d
    #print(v2.shape)
    #print(np.asarray(mask).shape)
    return v2,np.asarray(mask)


def protein_encoder(p):
    max_p = 545
    p_onebpe = p_bpe.process_line(p).split()
    try:
        v = np.asarray([p_wordtoidx[i] for i in p_onebpe])
    except:
        v = np.array([0])
        print(p)

    # mask
    length = len(v)
    if length<max_p:
        v2 = np.pad(v,(0,max_p-length),'constant',constant_values=0)
        mask = ([1]*length)+([0]*(max_p-length))
    else:
        # cut to maxlength
        v2 = v[:max_p]
        mask = [1]*max_p
    return v2,np.asarray(mask)
