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
from matchms.importing import load_from_mgf

# positive
spectrums = [s for s in load_from_mgf('Example/CASMI/all_casmi.mgf')]

# positive
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference = pickle.load(file)

outfile = os.path.join(path_data, 'In_House/ALL_Inhouse_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference += pickle.load(file)

outfile = os.path.join(path_data, 'NIST2020/ALL_NIST20_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference += pickle.load(file)

for s in tqdm(reference):
    if s.get('precursor_mz') is None:
        s.set('precursor_mz', 0)

pmz = np.array([s.get('precursor_mz') for s in tqdm(reference)])
reference = np.array(reference)[np.argsort(pmz)]

scores = calculate_scores(references=spectrums,
                          queries=reference,
                          similarity_function=CosineGreedy())

excludes = []
similarities = scores.scores
for i in tqdm(range(len(spectrums))):
    sim = np.array([s[0] for s in similarities[i,:]])
    wh = np.where(sim > 0.95)[0]
    excludes += list(wh)
    
new_reference = [reference[i] for i in tqdm(range(len(reference))) if i not in excludes]
pickle.dump(new_reference, 
            open(os.path.join('Saves/paper_version/references_spectrums_positive.pickle'), "wb"))


# negative
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference = pickle.load(file)

outfile = os.path.join(path_data, 'In_House/ALL_Inhouse_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference += pickle.load(file)

outfile = os.path.join(path_data, 'NIST2020/ALL_NIST20_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference += pickle.load(file)

for s in tqdm(reference):
    if s.get('precursor_mz') is None:
        s.set('precursor_mz', 0)

pmz = np.array([s.get('precursor_mz') for s in tqdm(reference)])
reference = np.array(reference)[np.argsort(pmz)]

scores = calculate_scores(references=spectrums,
                          queries=reference,
                          similarity_function=CosineGreedy())

excludes = []
similarities = scores.scores
for i in tqdm(range(len(spectrums))):
    sim = np.array([s[0] for s in similarities[i,:]])
    wh = np.where(sim > 0.95)[0]
    excludes += list(wh)
    
new_reference = [reference[i] for i in tqdm(range(len(reference))) if i not in excludes]
pickle.dump(new_reference, 
            open(os.path.join('Saves/paper_version/references_spectrums_negative.pickle'), "wb"))
