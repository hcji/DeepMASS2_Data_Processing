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

with open('Saves/paper_version/references_spectrums_positive.pickle', 'rb') as file:
    reference = pickle.load(file)
    
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
with open('Saves/paper_version/references_spectrums_negative.pickle', 'rb') as file:
    reference = pickle.load(file)
    
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