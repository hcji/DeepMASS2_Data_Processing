# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:28:08 2022

@author: DELL
"""


import os
import random
import numpy as np
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from matchms.importing import load_from_mgf
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model

# positive
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums = pickle.load(file)

spectrums = [s for s in tqdm(spectrums)]
random.shuffle(spectrums)

reference = []
for s in tqdm(spectrums):
    if len(s.mz[s.mz <= 1000]) < 5:
        continue
    if 'smiles' not in list(s.metadata.keys()):
        continue
    if s.metadata['smiles'] == '':
        continue
    try:
        mol = Chem.MolFromSmiles(s.metadata['smiles'])
        wt = AllChem.CalcExactMolWt(mol)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        formula = AllChem.CalcMolFormula(mol)
        s = s.set('formula', formula)
        s = s.set('smiles', smi)
        s = s.set('parent_mass', wt)
    except:
        continue
    if 'ionmode' not in list(s.metadata.keys()):
        continue  
    if s.metadata['ionmode'] == 'positive':
        keys = [s for s in s.metadata.keys() if s in ['compound_name', 'formula', 'smiles', 'inchikey', 'precursor_mz', 'adduct', 'parent_mass', 'ionmode', 'charge']]
        s.metadata = {k: s.metadata[k] for k in keys}
        reference.append(s)

pickle.dump(reference, open('Saves/public_version/references_spectrums_positive.pickle', "wb"))


with open('Saves/public_version/references_spectrums_positive.pickle', 'rb') as file:
    spectrums = pickle.load(file)

documents = [SpectrumDocument(s, n_decimals=2) for s in spectrums]
model_file = "Models/Ms2Vec_allGNPSpositive.hdf5"
iterations = [30]
model = train_new_word2vec_model(documents, iterations=iterations, filename=model_file,
                                 workers=8, progress_logger=True)




# negative
outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums = pickle.load(file)

spectrums = [s for s in tqdm(spectrums)]
random.shuffle(spectrums)

reference = []
for s in tqdm(spectrums):
    if len(s.mz[s.mz <= 1000]) < 5:
        continue
    if 'smiles' not in list(s.metadata.keys()):
        continue
    if s.metadata['smiles'] == '':
        continue
    try:
        mol = Chem.MolFromSmiles(s.metadata['smiles'])
        wt = AllChem.CalcExactMolWt(mol)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        formula = AllChem.CalcMolFormula(mol)
        s = s.set('formula', formula)
        s = s.set('smiles', smi)
        s = s.set('parent_mass', wt)
    except:
        continue
    if 'ionmode' not in list(s.metadata.keys()):
        continue
    if s.metadata['ionmode'] == 'negative':
        keys = [s for s in s.metadata.keys() if s in ['compound_name', 'formula', 'smiles', 'inchikey', 'precursor_mz', 'adduct', 'parent_mass', 'ionmode', 'charge']]
        s.metadata = {k: s.metadata[k] for k in keys}
        reference.append(s)
pickle.dump(reference, open('Saves/public_version/references_spectrums_negative.pickle', "wb"))


with open('Saves/public_version/references_spectrums_negative.pickle', 'rb') as file:
    spectrums = pickle.load(file)

documents = [SpectrumDocument(s, n_decimals=2) for s in spectrums]
model_file = "Models/Ms2Vec_allGNPSnegative.hdf5"
iterations = [30]
model = train_new_word2vec_model(documents, iterations=iterations, filename=model_file,
                                 workers=8, progress_logger=True)