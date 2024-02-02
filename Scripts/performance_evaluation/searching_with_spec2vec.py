# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:21:44 2023

@author: DELL
"""


import gensim
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from matchms import calculate_scores
from matchms.similarity import CosineGreedy
from matchms.importing import load_from_mgf
from spec2vec import SpectrumDocument
from spec2vec import Spec2Vec


with open('D:/DeepMASS2_GUI_20231025/DeepMASS2_GUI/data/references_spectrums_positive.pickle', 'rb') as file:
    reference_pos = pickle.load(file)
    
with open('D:/DeepMASS2_GUI_20231025/DeepMASS2_GUI/data/references_spectrums_negative.pickle', 'rb') as file:
    reference_neg = pickle.load(file)

reference_pos = np.array([s for s in reference_pos if Chem.MolFromSmiles(s.get('smiles')) is not None])
reference_neg = np.array([s for s in reference_neg if Chem.MolFromSmiles(s.get('smiles')) is not None])

reference_formula_pos = np.array([AllChem.CalcMolFormula(Chem.MolFromSmiles(s.get('smiles'))) for s in tqdm(reference_pos)])
reference_formula_neg = np.array([AllChem.CalcMolFormula(Chem.MolFromSmiles(s.get('smiles'))) for s in tqdm(reference_neg)])

reference_documents_pos = np.array([SpectrumDocument(s, n_decimals=2) for s in tqdm(reference_pos)])
reference_documents_neg = np.array([SpectrumDocument(s, n_decimals=2) for s in tqdm(reference_neg)])

model_pos = gensim.models.Word2Vec.load("D:/DeepMASS2_GUI_20231025/DeepMASS2_GUI/model/Ms2Vec_allGNPSnegative.hdf5")
model_neg = gensim.models.Word2Vec.load("D:/DeepMASS2_GUI_20231025/DeepMASS2_GUI/model/Ms2Vec_allGNPSpositive.hdf5")

spec2vec_similarity_pos = Spec2Vec(model=model_pos, intensity_weighting_power=1, allowed_missing_percentage=100)
spec2vec_similarity_neg = Spec2Vec(model=model_neg, intensity_weighting_power=1, allowed_missing_percentage=100)

output = []
spectrums = [s for s in load_from_mgf('Example/CASMI/all_casmi.mgf')]
for i,s in enumerate(tqdm(spectrums)):
    query_mode = s.get('ionmode')
    query_formula = s.get('formula')
    query_inchikey = s.get('inchikey')[:14]
    if query_mode == 'negative':
        keep = np.where(reference_formula_neg == query_formula)[0]
        reference_i = reference_neg[keep]
        reference_i_spec2vec = reference_documents_neg[keep]
        spec2vec_similarity = spec2vec_similarity_neg
    else:
        keep = np.where(reference_formula_pos == query_formula)[0]
        reference_i = reference_pos[keep]
        reference_i_spec2vec = reference_documents_pos[keep]
        spec2vec_similarity = spec2vec_similarity_pos
    
    all_inchikey = np.array([r.get('inchikey')[:14] for r in reference_i])
    
    if (len(reference_i) == 0) or (query_inchikey not in list(all_inchikey)):
        rank_spec2vec = float('inf')
        rank_cosine_greedy = float('inf')
        output.append([i, rank_cosine_greedy, rank_spec2vec])
        continue

    # spec2vec
    scores_spec2vec = calculate_scores(references=reference_i_spec2vec, queries=[s],
                                       similarity_function=spec2vec_similarity)
    scores_spec2vec = np.array([s[0] for s in scores_spec2vec.scores.to_array()])
    if max(scores_spec2vec) > 0:
        ranked_inchikey_spec2vec = all_inchikey[np.argsort(-scores_spec2vec)]
        rank_spec2vec = np.where(ranked_inchikey_spec2vec == query_inchikey)[0][0]
    else:
        rank_spec2vec = float('inf')
    
    # cosine
    scores_cosine_greedy = calculate_scores(references=reference_i, queries=[s], similarity_function=CosineGreedy())
    scores_cosine_greedy = np.array([s[0].tolist()[0] for s in scores_cosine_greedy.scores.to_array()])
    if max(scores_cosine_greedy) > 0:
        ranked_inchikey_cosine_greedy = all_inchikey[np.argsort(-scores_cosine_greedy)]
        rank_cosine_greedy = np.where(ranked_inchikey_cosine_greedy == query_inchikey)[0][0]
    else:
        rank_cosine_greedy = float('inf')
    
    output.append([i, rank_cosine_greedy, rank_spec2vec])
    
output = pd.DataFrame(output,
                      columns=['Spectrum Index', 'Greedy Cosine', 'Spec2Vec'])
    

# existed in reference
output = output[output['Greedy Cosine'] <= 9999]

ratios = []
for i in range(1, 11):
    cosine_ratio = len(np.where(output['Greedy Cosine'] <= i )[0]) / len(output)
    spec2vec_ratio = len(np.where(output['Spec2Vec'] <= i )[0]) / len(output)
    ratios.append([cosine_ratio, spec2vec_ratio])
ratios = pd.DataFrame(ratios, columns = ['Greedy Cosine', 'Spec2Vec'])
    
