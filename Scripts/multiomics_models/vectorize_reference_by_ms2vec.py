# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 08:46:57 2022

@author: DELL
"""

import os
import random
import numpy as np
import pickle
import hnswlib
import gensim
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from matchms.importing import load_from_mgf
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector


# positive
file = 'Models/Ms2Vec_allGNPSpositive.hdf5'
model = gensim.models.Word2Vec.load(file)
calc_ms2vec_vector = lambda x: calc_vector(model, SpectrumDocument(x, n_decimals=2), allowed_missing_percentage=100)

with open('Saves/multiomics/references_spectrums_positive.pickle', 'rb') as file:
    reference = pickle.load(file)

reference_vector = []
for s in tqdm(reference):
    reference_vector.append(calc_ms2vec_vector(s))

xb = np.array(reference_vector).astype('float32')
xb_len =  np.linalg.norm(xb, axis=1, keepdims=True)
xb = xb/xb_len
dim = 300
num_elements = len(xb)
ids = np.arange(num_elements)

p = hnswlib.Index(space = 'l2', dim = dim)
p.init_index(max_elements = num_elements, ef_construction = 800, M = 64)
p.add_items(xb, ids)
p.set_ef(300)
p.save_index('Saves/multiomics/references_index_positive_spec2vec.bin')

