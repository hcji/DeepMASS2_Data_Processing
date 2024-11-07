# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:12:47 2024

@author: DELL
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
from matchms.importing import load_from_mgf


database = pd.read_csv("D:/All_Database/database.csv")
with open('Saves/paper_version/references_spectrums_positive.pickle', 'rb') as file:
    reference = pickle.load(file)
with open('Saves/paper_version/references_spectrums_negative.pickle', 'rb') as file:
    reference += pickle.load(file)

reference_inchikeys = [s.get('inchikey') for s in tqdm(reference)]
reference_inchikeys = list(set([s for s in reference_inchikeys if s is not None]))
reference_inchikeys = list([s for s in reference_inchikeys if s != ''])

reference_classes = []
for inchikey in tqdm(reference_inchikeys):
    try:
        w = np.where(database['Short InChIKey'] == inchikey[:14])[0][0]
        cla = database.loc[w, 'Super Class']
        if cla is not np.nan:
            reference_classes.append(cla)
    except:
        continue
reference_classes_count = Counter(reference_classes)
reference_classes_count = pd.DataFrame({'class': reference_classes_count.keys(), 'number': reference_classes_count.values()})

database_classes = [s for s in database['Super Class'] if s is not np.nan]
database_classes_count = Counter(database_classes)
database_classes_count = pd.DataFrame({'class': database_classes_count.keys(), 'number': database_classes_count.values()})


casmi_smiles = [s.get('smiles') for s in load_from_mgf('Example/CASMI/all_casmi.mgf')]
anicancer_smiles = [s.get('smiles') for s in load_from_mgf('Example/Anticancer/export.mgf')]

database_smiles = [s for s in database['SMILES'] if s is not np.nan]
reference_smiles = [s.get('smiles') for s in tqdm(reference)]
reference_smiles = list(set([s for s in reference_smiles if s is not None]))
reference_smiles = list([s for s in reference_smiles if s != ''])


database_fps = []
for smi in tqdm(database_smiles):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024)
        database_fps.append(np.array(fp))
    except:
        continue
database_fps = np.array(database_fps)


reference_fps = []
for smi in tqdm(reference_smiles):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024)
        reference_fps.append(np.array(fp))
    except:
        continue
reference_fps = np.array(reference_fps)

casmi_fps = []
for smi in tqdm(casmi_smiles):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024)
        casmi_fps.append(np.array(fp))
    except:
        continue
casmi_fps = np.array(casmi_fps)

anicancer_fps = []
for smi in tqdm(anicancer_smiles):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024)
        anicancer_fps.append(np.array(fp))
    except:
        continue
anicancer_fps = np.array(anicancer_fps)


import umap
import umap.plot
import seaborn as sns

labels = np.array(['Test Compounds'] * len(casmi_fps) + ['Training Compounds'] * len(reference_fps))

fingerprints = np.vstack([np.vstack(casmi_fps), np.vstack(reference_fps)])
umap_embedder = umap.UMAP(n_neighbors=10, metric='jaccard').fit(fingerprints)
fingerprints_umap = umap_embedder.transform(fingerprints)

plt.figure(figsize=(5,4), dpi=300)
plt.scatter(fingerprints_umap[labels == 'Training Compounds', 0], 
            fingerprints_umap[labels == 'Training Compounds', 1], 
            color = 'lightskyblue', alpha = 0.5, marker = ".", label = 'Training Compounds')
plt.scatter(fingerprints_umap[labels == 'Test Compounds', 0], 
            fingerprints_umap[labels == 'Test Compounds', 1], 
            color = 'tomato', alpha = 0.5, marker = ".", label = 'CASMI Compounds')
plt.ylim(-21, 30)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(loc = 'upper left')


labels = np.array(['Test Compounds'] * len(anicancer_fps) + ['Training Compounds'] * len(reference_fps))

fingerprints = np.vstack([np.vstack(anicancer_fps), np.vstack(reference_fps)])
umap_embedder = umap.UMAP(n_neighbors=10, metric='jaccard').fit(fingerprints)
fingerprints_umap = umap_embedder.transform(fingerprints)

plt.figure(figsize=(5,4), dpi=300)
plt.scatter(fingerprints_umap[labels == 'Training Compounds', 0], 
            fingerprints_umap[labels == 'Training Compounds', 1], 
            color = 'lightskyblue', alpha = 0.5, marker = ".", label = 'Training Compounds')
plt.scatter(fingerprints_umap[labels == 'Test Compounds', 0], 
            fingerprints_umap[labels == 'Test Compounds', 1], 
            color = 'tomato', alpha = 0.5, marker = ".", label = 'Natural Products')
plt.ylim(-21, 30)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(loc = 'upper left')


k1 = np.where([(';;C' in k) or ((k[0] == 'C') and k[:2] != 'CH') for k in database['Database IDs']])[0]
labels = np.array(['Test Compounds'] * len(k1) + ['Training Compounds'] * len(reference_fps))

fingerprints = np.vstack([np.vstack(database_fps[k1,:]), np.vstack(reference_fps)])
umap_embedder = umap.UMAP(n_neighbors=10, metric='jaccard').fit(fingerprints)
fingerprints_umap = umap_embedder.transform(fingerprints)

plt.figure(figsize=(5,4), dpi=300)
plt.scatter(fingerprints_umap[labels == 'Training Compounds', 0], 
            fingerprints_umap[labels == 'Training Compounds', 1], 
            color = 'lightskyblue', alpha = 0.5, marker = ".", label = 'Training Compounds')
plt.scatter(fingerprints_umap[labels == 'Test Compounds', 0], 
            fingerprints_umap[labels == 'Test Compounds', 1], 
            color = 'peachpuff', alpha = 0.2, marker = ".", label = 'KEGG database')
plt.ylim(-21, 30)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(loc = 'upper left')


import json
import requests
from bs4 import BeautifulSoup
from scipy.stats import median_abs_deviation

casmi_results = pd.read_csv('Example/CASMI/ranking_result.csv')
anticancer_results = pd.read_csv('Example/Anticancer/ranking_result.csv')
all_results = pd.concat([casmi_results, anticancer_results], ignore_index=True)

def predict_class(smi, timeout = 60):
    url = 'https://npclassifier.ucsd.edu/classify?smiles={}'.format(smi)
    try:
        response = requests.get(url, timeout=timeout)
        soup = BeautifulSoup(response.content, "html.parser") 
        sub_class = json.loads(str(soup))['class_results']
        super_class = json.loads(str(soup))['superclass_results']
    except:
        return None
    if len(sub_class) >= 1:
        return {'class': sub_class[0], 'super_class': super_class[0]}
    else:
        return None

Class, SuperClass = [], []
for i in tqdm(all_results.index):
    smi = database.loc[i,'SMILES']
    res = predict_class(smi)
    if res is None:
        Class.append('')
        SuperClass.append('')
    else:
        Class.append(res['class'])
        SuperClass.append(res['super_class'])
all_results['Super Class'] = SuperClass


fingerprints = np.vstack(reference_fps)
umap_embedder = umap.UMAP(n_neighbors=10, metric='jaccard').fit(fingerprints)
fingerprints_umap = umap_embedder.transform(fingerprints)
test_fingerprints_umap = umap_embedder.transform(np.vstack([casmi_fps, anicancer_fps]))

center = np.mean(fingerprints_umap, axis = 0)
umap_dist = []
for i in tqdm(all_results.index):
    a = (test_fingerprints_umap[i, 0]-center[0])**2
    b = (test_fingerprints_umap[i, 1]-center[1])**2
    dist = np.sqrt(a + b)
    umap_dist.append(dist)
all_results['UMAP Dist'] = umap_dist


mol_mass = [AllChem.CalcExactMolWt(Chem.MolFromSmiles(s)) for s in all_results['SMILES']]
all_results['Mol Mass'] = mol_mass


# print(Counter(SuperClass))
classes = ['Small peptides', 'Flavonoids', 'Steroids', 'Phenylpropanoids (C6-C3)', 'Diterpenoids', 'Saccharides']
pltdata = []
for cla in classes:
    w = np.where(all_results['Super Class'] == cla)[0]
    m = ['DeepMASS Ranking', 'SIRIUS Ranking','MSFinder Ranking', 'MetFrag Ranking', 'CFM-ID Ranking']
    for ww in w:
        for mm in m:
            pltdata.append([cla, mm, np.nanmin([10, all_results.loc[ww, mm]])])
pltdata = pd.DataFrame(pltdata, columns=['class', 'method', 'rank'])
plt.figure(figsize=(7,4), dpi=300)
sns.violinplot(x="class", y="rank", hue="method", data=pltdata, inner='quart', palette='husl')
plt.xlabel("Molecular Class")
plt.xticks(ticks=range(6), labels=['Small peptide', 'Flavonoid', 'Steroid', 'Phenylprop', 'Diterpenoid', 'Saccharide'])
plt.ylabel("Ranking")
'''
handles, labels = plt.gca().get_legend_handles_labels()
labels = ['DeepMASS', 'SIRIUS','MSFinder', 'MetFrag', 'CFM-ID']
n = len(set(pltdata['method']))  # Number of categories
plt.legend(handles[:n], labels[:n], title="Category", loc="upper center", 
           bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=5)
'''
plt.show()

quantiles = [np.quantile(all_results['UMAP Dist'], i) for i in np.arange(0,1.1,0.2)]
classes = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
pltdata = []
for i, cla in enumerate(classes):
    w = np.where(np.logical_and(all_results['UMAP Dist'] >= quantiles[i], all_results['UMAP Dist'] <= quantiles[i+1]))[0]
    m = ['DeepMASS Ranking', 'SIRIUS Ranking','MSFinder Ranking', 'MetFrag Ranking', 'CFM-ID Ranking']
    for ww in w:
        for mm in m:
            pltdata.append([cla, mm, np.nanmin([10, all_results.loc[ww, mm]])])
pltdata = pd.DataFrame(pltdata, columns=['class', 'method', 'rank'])
plt.figure(figsize=(7,4), dpi=300)
sns.violinplot(x="class", y="rank", hue="method", data=pltdata, inner='quart', palette='husl', legend=False)
plt.xlabel("UMAP Distance Quantiles")
plt.ylabel("Ranking")
plt.show()


massrange = [0, 200, 400, 600, 800, 1600]
classes = ['< 200', '200-400', '400-600', '600-800', '>800']
pltdata = []
for i, cla in enumerate(classes):
    w = np.where(np.logical_and(all_results['Mol Mass'] >= massrange[i], all_results['Mol Mass'] <= massrange[i+1]))[0]
    m = ['DeepMASS Ranking', 'SIRIUS Ranking','MSFinder Ranking', 'MetFrag Ranking', 'CFM-ID Ranking']
    for ww in w:
        for mm in m:
            pltdata.append([cla, mm, np.nanmin([10, all_results.loc[ww, mm]])])
pltdata = pd.DataFrame(pltdata, columns=['class', 'method', 'rank'])
plt.figure(figsize=(7,4), dpi=300)
sns.violinplot(x="class", y="rank", hue="method", data=pltdata, inner='quart', palette='husl', legend=False)
plt.xlabel("Molecular Mass")
plt.ylabel("Ranking")
plt.show()