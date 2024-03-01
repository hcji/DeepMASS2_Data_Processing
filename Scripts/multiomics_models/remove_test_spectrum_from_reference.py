# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:30:51 2023

@author: DELL
"""


import os
import pickle
import numpy as np
from tqdm import tqdm
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
from matchms.importing import load_from_msp

spectrums = [s for s in load_from_msp('Example/ProtAssociated/gene_associated_spectrums.msp')]

# positive
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference = pickle.load(file)

pmz = np.array([s.get('precursor_mz') for s in tqdm(reference)])
reference = np.array(reference)[np.argsort(pmz)]

excludes = []
for i in tqdm(range(len(spectrums))):
    scores = calculate_scores(references=reference,
                            queries=[spectrums[i]],
                            similarity_function=CosineGreedy())
    scores = scores.scores.to_array()
    sims = np.array([s[0][0] for s in scores])
    wh = np.where(sims > 0.95)[0]
    excludes += list(wh)
    
new_reference = [reference[i] for i in tqdm(range(len(reference))) if i not in excludes]
pickle.dump(new_reference, 
            open(os.path.join('Saves/multiomics/references_spectrums_positive.pickle'), "wb"))

