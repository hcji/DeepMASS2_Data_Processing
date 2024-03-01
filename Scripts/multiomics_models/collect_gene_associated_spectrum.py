# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:01:30 2023

@author: DELL
"""


import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import inchi
from matchms.exporting import save_as_msp

metabolite_protein_associations = pd.read_pickle('Datasets/HMDB/metabolite_gene_associations.pickle')

# positive
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference = pickle.load(file)

np.random.shuffle(reference)

metabolite_inchikeys, gene_ids = [], []
for i in tqdm(metabolite_protein_associations.index):
    smiles = metabolite_protein_associations.loc[i, 'Association SMILES']
    protid = metabolite_protein_associations.loc[i, 'Uniprot ID']
    geneid = metabolite_protein_associations.loc[i, 'Gene Name']
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            inchikey = inchi.MolToInchiKey(mol)
        except:
            continue
        metabolite_inchikeys.append(inchikey)
        gene_ids.append(geneid)
metabolite_inchikeys = np.array(metabolite_inchikeys)
gene_ids = np.array(gene_ids)

existed_inchikeys = []
gene_associated_spectrums = []
for s in tqdm(reference):
    inchikey = s.get('inchikey')
    if inchikey in existed_inchikeys:
        continue
    if inchikey in metabolite_inchikeys:
        gene_id = gene_ids[np.where(metabolite_inchikeys==inchikey)[0]]
        gene_id = [g for g in gene_id if g is not None]
        if len(gene_id) == 0:
            continue
        s = s.set('associated_gene', ','.join(gene_id))
        existed_inchikeys.append(inchikey)
        gene_associated_spectrums.append(s)
save_as_msp(gene_associated_spectrums, 'Example/ProtAssociated/gene_associated_spectrums.msp')
