# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:28:08 2022

@author: DELL
"""


import random
import pickle
import gensim
import numpy as np

from tqdm import tqdm
from matchms.filtering import add_losses
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model


settings = {
    "sg": 0,
    "negative": 5,
    "vector_size": 300,
    "window": 500,
    "min_count": 1,
    "workers": 8,
    "compute_loss": True,
}


class SpectrumDocumentDev(SpectrumDocument):
    def __init__(self, s, n_decimals: int = 2):
        self.s = s
        self.mz = s.mz
        self.intensities = s.intensities
        super().__init__(s, n_decimals)
        self._make_words_with_deviation()
        
    def _make_words_with_deviation(self, ppm = 5):     
        dev = np.array([np.random.normal(0, m) for m in self.mz * ppm / 10 **6])
        peak_words = [f"peak@{mz:.{self.n_decimals}f}" for mz in self.mz + dev]
        self.words = peak_words


# positive
with open('Saves/paper_version/references_spectrums_positive.pickle', 'rb') as file:
    spectrums = pickle.load(file)

random.shuffle(spectrums)
documents = [SpectrumDocumentDev(s, n_decimals=2) for s in tqdm(spectrums)]

epochs = 200
model = gensim.models.Word2Vec(documents, epochs=1, **settings)
for epoch in range(epochs):
    print('start epoch of {}'.format(epoch))
    random.shuffle(spectrums)
    documents = [SpectrumDocumentDev(s, n_decimals=2) for s in tqdm(spectrums)]
    model.build_vocab(documents, update=True)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    
model_file = "Models/ppm_10/Ms2Vec_allGNPSpositive.hdf5"
model.save(model_file)


# negative
with open('Saves/paper_version/references_spectrums_negative.pickle', 'rb') as file:
    spectrums = pickle.load(file)

random.shuffle(spectrums)
documents = [SpectrumDocumentDev(s, n_decimals=2) for s in tqdm(spectrums)]

epochs = 200
model = gensim.models.Word2Vec(documents, epochs=1, **settings)
for epoch in range(epochs):
    print('start epoch of {}'.format(epoch))
    random.shuffle(spectrums)
    documents = [SpectrumDocumentDev(s, n_decimals=2) for s in tqdm(spectrums)]
    model.build_vocab(documents, update=True)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    
model_file = "Models/ppm_10/Ms2Vec_allGNPSnegative.hdf5"
model.save(model_file)

