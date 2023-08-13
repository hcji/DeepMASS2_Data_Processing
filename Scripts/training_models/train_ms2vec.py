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

inchikey_test = np.load('Saves/paper_version/inchikey_test.npy', allow_pickle=True)
spectrums_casmi = [s for s in load_from_mgf('example/CASMI/all_casmi.mgf')]
inchikey_casmi = list(set([s.get('inchikey')[:14] for s in spectrums_casmi]))
inchikey_test = [s for s in inchikey_test if s not in inchikey_casmi]

# positive
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums = pickle.load(file)

outfile = os.path.join(path_data, 'In_House/ALL_Inhouse_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)

outfile = os.path.join(path_data, 'NIST2020/ALL_NIST20_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)

spectrums = [s for s in tqdm(spectrums)]
random.shuffle(spectrums)


reference, test = [], []
count_test = np.zeros(len(inchikey_test))
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
        inchikey = s.get('inchikey')[:14]
        if inchikey in inchikey_casmi:
            continue
        if inchikey in inchikey_test:
            i = inchikey_test.index(inchikey)
            if count_test[i] == 0:
                count_test[i] += 1
                test.append(s)
                continue
        reference.append(s)

pickle.dump(reference, open('Saves/paper_version/references_spectrums_positive.pickle', "wb"))
pickle.dump(test, open('Example/Test/test_spectrums_positive.pickle', "wb"))


with open('Saves/paper_version/references_spectrums_positive.pickle', 'rb') as file:
    spectrums = pickle.load(file)

documents = [SpectrumDocument(s, n_decimals=2) for s in spectrums]
model_file = "model/Ms2Vec_allGNPSpositive.hdf5"
iterations = [30]
model = train_new_word2vec_model(documents, iterations=iterations, filename=model_file,
                                 workers=8, progress_logger=True)




# negative
path_data = 'D:/All_MSDatabase'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums = pickle.load(file)

outfile = os.path.join(path_data, 'In_House/ALL_Inhouse_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)

outfile = os.path.join(path_data, 'NIST2020/ALL_NIST20_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)

spectrums = [s for s in tqdm(spectrums)]
random.shuffle(spectrums)


reference, test = [], []
count_test = np.zeros(len(inchikey_test))
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
        inchikey = s.get('inchikey')[:14]
        if inchikey in inchikey_casmi:
            continue
        if inchikey in inchikey_test:
            i = inchikey_test.index(inchikey)
            if count_test[i] == 0:
                count_test[i] += 1
                test.append(s)
                continue
        reference.append(s)
pickle.dump(reference, open('Saves/paper_version/references_spectrums_negative.pickle', "wb"))
pickle.dump(test, open('Example/Test/test_spectrums_negative.pickle', "wb"))


with open('Saves/paper_version/references_spectrums_negative.pickle', 'rb') as file:
    spectrums = pickle.load(file)

documents = [SpectrumDocument(s, n_decimals=2) for s in spectrums]
model_file = "model/Ms2Vec_allGNPSnegative.hdf5"
iterations = [30]
model = train_new_word2vec_model(documents, iterations=iterations, filename=model_file,
                                 workers=8, progress_logger=True)
