# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:13:11 2022

@author: DELL
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from rdkit import Chem
from matchms.importing import load_from_mgf

import gensim
import pickle
from rdkit.Chem import AllChem
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
from spec2vec import SpectrumDocument
from spec2vec import Spec2Vec


spectrums = [s for s in load_from_mgf('Example/CASMI/all_casmi.mgf')]
inchikeys = [s.get('inchikey') for s in spectrums]
print(len(set(inchikeys)))

sirius_path = "Example/CASMI/sirius"
sirius_files = [name for name in os.listdir(sirius_path) if os.path.isdir(os.path.join(sirius_path, name)) ]
sirius_index = [int(i.split('_')[-2]) for i in sirius_files]

deepmass_path = "Example/CASMI/result"
deepmass_files = [name for name in os.listdir(deepmass_path)]
deepmass_index = [int(i.split('_')[-1].split('.')[-2]) for i in deepmass_files]

matchms_path = "Example/CASMI/matchms"
matchms_files = [name for name in os.listdir(matchms_path)]
matchms_index = [int(i.split('_')[-1].split('.')[-2]) for i in deepmass_files]

msfinder_path = "Example/CASMI/msfinder/Structure result-2108.txt"
msfinder_result = pd.read_csv(msfinder_path, sep = '\t')
msfinder_columns = [col for col in msfinder_result.columns if 'InChIKey' in col]

deepmass_path_1 = "Example/CASMI/result_new"
deepmass_files_1 = [name for name in os.listdir(deepmass_path_1)]
deepmass_index_1 = [int(i.split('_')[-1].split('.')[-2]) for i in deepmass_files_1]


ranking_result = []
for s in tqdm(spectrums):
    name = s.metadata['compound_name']
    index = int(name.split('_')[-1])
    true_key = s.metadata['inchikey'][:14]
    
    # rank of sirius
    sirius_file = "/{}/structure_candidates.tsv".format(sirius_files[sirius_index.index(index)])
    sirius_file = sirius_path + sirius_file
    try:
        sirius_result = pd.read_csv(sirius_file, sep='\t')
    except:
        sirius_result = None

    if sirius_result is not None:
        sirius_n = len(sirius_result)
        sirius_key = np.array([k for k in sirius_result['InChIkey2D']])
        sirius_rank = np.where(sirius_key == true_key)[0]
        if len(sirius_rank) == 0:
            sirius_rank = float('inf')
        else:
            sirius_rank = sirius_rank[0] + 1
    else:
        sirius_n = 0
        sirius_rank = float('inf')
    
    # rank of deepmass open
    deepmass_file_1 = "/{}".format(deepmass_files_1[deepmass_index_1.index(index)])
    deepmass_file_1 = deepmass_path_1 + deepmass_file_1
    deepmass_result_1 = pd.read_csv(deepmass_file_1)
    deepmass_key_1 = np.array([k[:14] for k in deepmass_result_1['InChIKey']])
    deepmass_n_1 = len(deepmass_key_1)
    deepmass_rank_1 = np.where(deepmass_key_1 == true_key)[0]
    if len(deepmass_rank_1) == 0:
        deepmass_rank_1 = float('inf')
    else:
        deepmass_rank_1 = deepmass_rank_1[0] + 1


    # rank of deepmass
    deepmass_file = "/{}".format(deepmass_files[deepmass_index.index(index)])
    deepmass_file = deepmass_path + deepmass_file
    deepmass_result = pd.read_csv(deepmass_file)
    deepmass_key = np.array([k[:14] for k in deepmass_result['InChIKey']])
    deepmass_n = len(deepmass_key)
    deepmass_rank = np.where(deepmass_key == true_key)[0]
    if len(deepmass_rank) == 0:
        deepmass_rank = float('inf')
    else:
        deepmass_rank = deepmass_rank[0] + 1

        
    # rank of matchms
    matchms_file = "/{}".format(matchms_files[matchms_index.index(index)])
    matchms_file = matchms_path + matchms_file
    matchms_result = pd.read_csv(matchms_file)
    matchms_key = np.array([k[:14] for k in matchms_result['InChIKey']])
    matchms_n = len(matchms_key)
    matchms_rank = np.where(matchms_key == true_key)[0]
    if len(matchms_rank) == 0:
        matchms_rank = float('inf')
    else:
        matchms_rank = matchms_rank[0] + 1        
        
        
    # rank of ms-finder
    msfinder_index = np.where(msfinder_result['File name'].values == name)[0]
    if len(msfinder_index) > 0:
        msfinder_key = [str(s)[:14] for s in msfinder_result.loc[msfinder_index[0], msfinder_columns].values]
        msfinder_rank = np.where(np.array(msfinder_key) == true_key)[0]
        if len(msfinder_rank) == 0:
            msfinder_rank = float('inf')
        else:
            msfinder_rank = msfinder_rank[0] + 1
    else:
        msfinder_rank = np.nan

    ranking_result.append([name, true_key, sirius_rank, msfinder_rank, deepmass_rank, deepmass_rank_1, matchms_rank])

ranking_result = pd.DataFrame(ranking_result, columns = ['Challenge', 'True Inchikey2D', 'SIRIUS Ranking', 'MSFinder Ranking',
                                                         'DeepMASS Ranking', 'DeepMASS Ranking (Public)', 'MatchMS Ranking'])



# searching with spec2vec
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

spec2vec_rank = []
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
        spec2vec_rank.append(rank_spec2vec)
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
    
    spec2vec_rank.append(rank_spec2vec)

ranking_result['Spec2Vec'] = spec2vec_rank



# total result
ratios = []
for i in range(1, 11):
    deepmass_ratio = len(np.where(ranking_result['DeepMASS Ranking'] <= i )[0]) / len(ranking_result)
    deepmass_ratio_1 = len(np.where(ranking_result['DeepMASS Ranking (Public)'] <= i )[0]) / len(ranking_result)
    msfinder_ratio = len(np.where(ranking_result['MSFinder Ranking'] <= i )[0]) / len(ranking_result)
    sirius_ratio = len(np.where(ranking_result['SIRIUS Ranking'] <= i )[0]) / len(ranking_result)
    matchms_ratio = len(np.where(ranking_result['MatchMS Ranking'] <= i )[0]) / len(ranking_result)
    spec2vec_ratio = len(np.where(ranking_result['Spec2Vec'] <= i )[0]) / len(ranking_result)
    
    ratios.append([deepmass_ratio, deepmass_ratio_1, sirius_ratio, msfinder_ratio, matchms_ratio, spec2vec_ratio])
ratios = pd.DataFrame(ratios, columns = ['DeepMASS', 'DeepMASS (Public)', 'SIRIUS', 'MSFinder', 'MatchMS', 'Spec2Vec'])

x = np.arange(1,11)
plt.figure(dpi = 300, figsize=(4.8,4.2))
plt.plot(x, ratios['DeepMASS'], label = 'DeepMASS', marker='D', color = '#FA7F6F')
plt.plot(x, ratios['DeepMASS (Public)'], label = 'DeepMASS (Public)', marker='D', color = '#FA7F6F', linestyle = '--')
plt.plot(x, ratios['SIRIUS'], label = 'SIRIUS', marker='D', color = '#FFBE7A')
plt.plot(x, ratios['MSFinder'], label = 'MSFinder', marker='D', color = '#8ECFC9')
plt.plot(x, ratios['MatchMS'], label = 'Cosine', marker='D', color = '#82B0D2')
plt.plot(x, ratios['Spec2Vec'], label = 'Spec2Vec', marker='D', color = '#BEA6C3')
plt.xlim(0.5, 10.5)
plt.ylim(0, 0.75)
plt.xticks(np.arange(1, 11, 1))
plt.xlabel('topK', fontsize = 12)
plt.ylabel('ratio', fontsize = 12)
plt.legend(loc='lower right')


# existed in reference
ranking_result_1 = ranking_result[ranking_result['MatchMS Ranking'] <= 9999]

ratios = []
for i in range(1, 11):
    deepmass_ratio = len(np.where(ranking_result_1['DeepMASS Ranking'] <= i )[0]) / len(ranking_result_1)
    deepmass_ratio_1 = len(np.where(ranking_result_1['DeepMASS Ranking (Public)'] <= i )[0]) / len(ranking_result_1)
    msfinder_ratio = len(np.where(ranking_result_1['MSFinder Ranking'] <= i )[0]) / len(ranking_result_1)
    sirius_ratio = len(np.where(ranking_result_1['SIRIUS Ranking'] <= i )[0]) / len(ranking_result_1)
    matchms_ratio = len(np.where(ranking_result_1['MatchMS Ranking'] <= i )[0]) / len(ranking_result_1)
    spec2vec_ratio = len(np.where(ranking_result_1['Spec2Vec'] <= i )[0]) / len(ranking_result_1)
    
    ratios.append([deepmass_ratio, deepmass_ratio_1, sirius_ratio, msfinder_ratio, matchms_ratio, spec2vec_ratio])
ratios = pd.DataFrame(ratios, columns = ['DeepMASS', 'DeepMASS (Public)', 'SIRIUS', 'MSFinder', 'MatchMS', 'Spec2Vec'])

x = np.arange(1,11)
plt.figure(dpi = 300, figsize=(4.8,4.2))
plt.plot(x, ratios['DeepMASS'], label = 'DeepMASS', marker='D', color = '#FA7F6F')
plt.plot(x, ratios['DeepMASS (Public)'], label = 'DeepMASS (Public)', marker='D', color = '#FA7F6F', linestyle = '--')
plt.plot(x, ratios['SIRIUS'], label = 'SIRIUS', marker='D', color = '#FFBE7A')
plt.plot(x, ratios['MSFinder'], label = 'MSFinder', marker='D', color = '#8ECFC9')
plt.plot(x, ratios['MatchMS'], label = 'Cosine', marker='D', color = '#82B0D2')
plt.plot(x, ratios['Spec2Vec'], label = 'Spec2Vec', marker='D', color = '#BEA6C3')
plt.xlim(0.5, 10.5)
plt.ylim(0.6, 1.0)
plt.xticks(np.arange(1, 11, 1))
plt.xlabel('topK', fontsize = 12)
plt.ylabel('ratio', fontsize = 12)
plt.legend(loc='lower right')


# not existed in reference
ranking_result_2 = ranking_result[ranking_result['MatchMS Ranking'] >= 9999]

ratios = []
for i in range(1, 11):
    deepmass_ratio = len(np.where(ranking_result_2['DeepMASS Ranking'] <= i )[0]) / len(ranking_result_2)
    msfinder_ratio = len(np.where(ranking_result_2['MSFinder Ranking'] <= i )[0]) / len(ranking_result_2)
    sirius_ratio = len(np.where(ranking_result_2['SIRIUS Ranking'] <= i )[0]) / len(ranking_result_2)
    deepmass_ratio_1 = len(np.where(ranking_result_2['DeepMASS Ranking (Public)'] <= i )[0]) / len(ranking_result_2)
    
    ratios.append([deepmass_ratio, sirius_ratio, msfinder_ratio, deepmass_ratio_1])
ratios = pd.DataFrame(ratios, columns = ['DeepMASS', 'SIRIUS', 'MSFinder', 'DeepMASS (Public)'])

x = np.arange(1,11)
plt.figure(dpi = 300, figsize=(4.8,4.2))
plt.plot(x, ratios['DeepMASS'], label = 'DeepMASS', marker='D', color = '#FA7F6F')
plt.plot(x, ratios['DeepMASS (Public)'], label = 'DeepMASS (Public)', marker='D', color = '#FA7F6F', linestyle = '--')
plt.plot(x, ratios['SIRIUS'], label = 'SIRIUS', marker='D', color = '#FFBE7A')
plt.plot(x, ratios['MSFinder'], label = 'MSFinder', marker='D', color = '#8ECFC9')
plt.xlim(0.5, 10.5)
plt.ylim(0.15, 0.55)
plt.xticks(np.arange(1, 11, 1))
plt.xlabel('topK', fontsize = 12)
plt.ylabel('ratio', fontsize = 12)
plt.legend(loc='lower right')
