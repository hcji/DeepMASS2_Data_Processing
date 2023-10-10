# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:28:08 2022

@author: DELL
"""


import random
import pickle

from tqdm import tqdm
from matchms.filtering import add_losses
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model

# positive
with open('Saves/paper_version/references_spectrums_positive.pickle', 'rb') as file:
    spectrums = pickle.load(file)

random.shuffle(spectrums)
# spectrums = [add_losses(s) for s in tqdm(spectrums)]
documents = [SpectrumDocument(s, n_decimals=2) for s in spectrums]
model_file = "Models/Ms2Vec_allGNPSpositive.hdf5"
iterations = [30]
model = train_new_word2vec_model(documents, iterations=iterations, filename=model_file, workers=8, progress_logger=True)


# negative
with open('Saves/paper_version/references_spectrums_negative.pickle', 'rb') as file:
    spectrums = pickle.load(file)

random.shuffle(spectrums)
# spectrums = [add_losses(s) for s in tqdm(spectrums)]
documents = [SpectrumDocument(s, n_decimals=2) for s in spectrums]
model_file = "Models/Ms2Vec_allGNPSnegative.hdf5"
iterations = [30]
model = train_new_word2vec_model(documents, iterations=iterations, filename=model_file, workers=8, progress_logger=True)
