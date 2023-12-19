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
from matchms.exporting import save_as_mgf

metabolite_protein_associations = np.load('Datasets/HMDB/metabolite_protein_associations.npy', allow_pickle=True)
metabolite_protein_associations = pd.DataFrame(metabolite_protein_associations, columns=['Uniprot ID', 'HMDB ID', 'Sequence', 'Association SMILES'])

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

metabolite_inchikeys, protein_ids = [], []
for i in tqdm(metabolite_protein_associations.index):
    smiles = metabolite_protein_associations.loc[i, 'Association SMILES']
    protid = metabolite_protein_associations.loc[i, 'Uniprot ID']
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            inchikey = inchi.MolToInchiKey(mol)
        except:
            continue
        metabolite_inchikeys.append(inchikey)
        protein_ids.append(protid)
metabolite_inchikeys = np.array(metabolite_inchikeys)
protein_ids = np.array(protein_ids)


prot_associated_spectrums = []
for s in tqdm(reference):
    inchikey = s.get('inchikey')
    if inchikey in metabolite_inchikeys:
        protid = protein_ids[np.where(metabolite_inchikeys==inchikey)[0]]
        s = s.set('associated_protein', ','.join(protid))
        prot_associated_spectrums.append(s)
save_as_mgf(prot_associated_spectrums, 'Example/ProtAssociated/prot_associated_spectrums.mgf')
