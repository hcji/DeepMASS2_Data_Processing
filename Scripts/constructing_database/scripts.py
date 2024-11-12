# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:24:10 2023

@author: DELL
"""


import os
import re
import json
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import inchi, AllChem
from molmass import Formula
from bs4 import BeautifulSoup


molTitle = []
molInChIkey = []
molShortkey = []
molExactmass = []
molFormula = []
molSMILES = []
molDatabaseID = []


print('11. PMN...')
for f in tqdm(os.listdir('PMN')):
    data = pd.read_csv('PMN/{}'.format(f), sep = '\t')
    for i in data.index:
        smiles = str(data.loc[i, 'Smiles'])
        if '*' in smiles:
            continue        
        try:
            m = Chem.MolFromSmiles(data.loc[i, 'Smiles'])
        except:
            continue
        if m is None:
            continue
        try:
            title = data.loc[i, 'Compound_common_name']
            inchikey = inchi.MolToInchiKey(m)
            dbid = 'PMN:{}'.format(data.loc[i, 'Compound_id'])
            if inchikey in molInChIkey:
                w = molInChIkey.index(inchikey)
                if dbid not in molDatabaseID[w]:
                    molDatabaseID[w].append(dbid)
                continue
            shortkey = inchikey[:14]
            molwt = AllChem.CalcExactMolWt(m)
            formula = AllChem.CalcMolFormula(m)
            smiles = Chem.MolToSmiles(m)
        except:
            continue

        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])


print('1. ChEBI ...')
mols = Chem.SDMolSupplier('ChEBI/ChEBI_lite_3star.sdf')
for m in tqdm(mols):
    if m is None:
        continue
    try:
        title = m.GetProp('ChEBI Name')
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = m.GetProp('ChEBI ID')
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)

        
print('2. BloodExp...')
data = pd.read_csv('BloodExp/blood_exposome_chemicals_july_2023.csv')
for i in tqdm(data.index):
    m = Chem.MolFromSmiles(data.loc[i, 'CanonicalSMILES'])
    if m is None:
        continue
    try:
        title = data.loc[i, 'Title']
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = 'BloodExp:CID{}'.format(data.loc[i, 'CID'])
    except:
        continue

    if '*' in smiles:
        continue
        
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)
        
        
print('3. DrugBank...')
mols = Chem.SDMolSupplier('DrugBank/structures.sdf')
for m in tqdm(mols):
    if m is None:
        continue
    try:
        title = m.GetProp('GENERIC_NAME')
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = m.GetProp('DATABASE_ID')
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('4. ECMDB...')
mols = json.load(open('ECMDB/ecmdb.json', encoding='utf-8'))
for mol in tqdm(mols):
    if mol['moldb_smiles'] is None:
        continue
    m = Chem.MolFromSmiles(mol['moldb_smiles'])
    if m is None:
        continue
    try:
        title = mol['biocyc_id']
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = mol.GetProp('met_id')
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)
        
        
print('5. FooDB...')
data = pd.read_csv('FooDB/FooDB_Compound.csv')
for i in tqdm(data.index):
    try:
        m = Chem.MolFromInchi(data.loc[i, 'moldb_inchikey'])
    except:
        continue
    if m is None:
        continue
    try:
        title = data.loc[i, 'name']
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = data.loc[i, 'public_id']
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('6. HMDB...')
mols = Chem.SDMolSupplier('HMDB/structures.sdf')
for m in tqdm(mols):
    if m is None:
        continue
    try:
        title = m.GetProp('GENERIC_NAME')
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = m.GetProp('DATABASE_ID')
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('7. KEGG...')
data = pd.read_csv('KEGG/kegg_compounds.csv')
for i in tqdm(data.index):
    try:
        m = Chem.MolFromSmiles(data.loc[i, 'smile'])
    except:
        continue
    if m is None:
        continue
    try:
        title = data.loc[i, 'Name']
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = data.loc[i, 'ID']
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('8. NANPDB...')
data = pd.read_csv('NANPDB/smiles_unique_all.smi', header=None, sep = '\t')
for i in tqdm(data.index):
    try:
        m = Chem.MolFromSmiles(data.iloc[i, 0])
    except:
        continue
    if m is None:
        continue
    try:
        title = data.iloc[i, 1]
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = 'NANPDB:{}'.format(i)
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('9. NPAtlas...')
data = pd.read_csv('NPAtlas/NPAtlas_download.tsv', sep = '\t')
for i in tqdm(data.index):
    try:
        m = Chem.MolFromSmiles(data.loc[i, 'compound_smiles'])
    except:
        continue
    if m is None:
        continue
    try:
        title = data.loc[i, 'compound_names']
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = data.loc[i, 'npaid']
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('10. PMHub...')
data = pd.read_csv('PMHub/metabolite_id.txt', sep = '\t', header = None)
for i in tqdm(data.index):
    try:
        m = Chem.MolFromSmiles(data.iloc[i, 1])
    except:
        continue
    if m is None:
        continue
    try:
        title = data.iloc[i, 5]
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = 'PMHub:{}'.format(data.iloc[i, 0])
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('12. SMPDB...')
for f in tqdm(os.listdir('SMPDB')):
    mols = Chem.SDMolSupplier('SMPDB/{}'.format(f))
    for m in mols:
        if m is None:
            continue
        try:
            title = m.GetProp('GENERIC_NAME')
            inchikey = inchi.MolToInchiKey(m)
            dbid = m.GetProp('DATABASE_ID')
            if inchikey in molInChIkey:
                w = molInChIkey.index(inchikey)
                if dbid not in molDatabaseID[w]:
                    molDatabaseID[w].append(dbid)
                continue            
            shortkey = inchikey[:14]
            molwt = AllChem.CalcExactMolWt(m)
            formula = AllChem.CalcMolFormula(m)
            smiles = Chem.MolToSmiles(m)
        except:
            continue
        
        if '*' in smiles:
            continue
        
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])


print('13. STOFF...')
mols = Chem.SDMolSupplier('STOFF/CCD-Batch-Search_2023-11-09_01_14_51.sdf')
for m in tqdm(mols):
    if m is None:
        continue
    try:
        title = m.GetProp('PREFERRED_NAME')
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = m.GetProp('DTXSID')
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('14. T3DB...')
mols = Chem.SDMolSupplier('T3DB/structures.sdf')
for m in tqdm(mols):
    if m is None:
        continue
    try:
        title = m.GetProp('NAME')
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = m.GetProp('DATABASE_ID')
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('15. YMDB')
mols = Chem.SDMolSupplier('YMDB/ymdb.sdf')
for m in tqdm(mols):
    if m is None:
        continue
    try:
        title = m.GetProp('GENERIC_NAME')
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = m.GetProp('DATABASE_ID')
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('16. NIST')
path = "D:/DeepMASS2_Data_Processing/Datasets/NIST2023/preprocessed_spectrums.npy"
mols = np.load(path, allow_pickle=True)
for mol in tqdm(mols):
    inchikey = str(mol.get('inchikey'))
    if inchikey in molInChIkey:
        continue
    m = Chem.MolFromSmiles(str(mol.get('smiles')))
    if m is None:
        continue
    try:
        title = mol.get('compound_name')
        inchikey = inchi.MolToInchiKey(m)
        if inchikey in molInChIkey:
            continue
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = 'NIST:CAS{}'.format(mol.get('casno'))
    except:
        continue
    
    if '*' in smiles:
        continue
    
    molTitle.append(title)
    molInChIkey.append(inchikey)
    molShortkey.append(shortkey)
    molExactmass.append(molwt)
    molFormula.append(formula)
    molSMILES.append(smiles)
    molDatabaseID.append([dbid])


print('17. TCMSP')
mols = np.load('TCMSP/TCMSP_Database.npy', allow_pickle=True)
for mol in tqdm(mols):
    if 'smiles' not in mol.keys():
        continue
    smi = mol['smiles']
    m = Chem.MolFromSmiles(mol.get('smiles'))
    if m is None:
        continue
    try:
        title = mol['molecule_name']
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = 'TCMSP:{}'.format(mol['MOL_ID'])
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)


print('18. GNPS')
path_data = 'D:/DeepMASS2_Data_Processing/Datasets/GNPS_clean'
spectrums = np.load(os.path.join(path_data, 'preprocessed_spectrums.npy'), allow_pickle=True)
gnps_index =  1
for s in tqdm(spectrums):
    smi = s.get('smiles')
    title = s.get('compound_name')
    title = str(re.sub(r'_\d+$', '', title))
    
    if bool(re.search(r'\d{3,}', title)):
        continue
    
    if smi is None:
        continue
    if smi == '':
        continue
    if '.' in smi:
        continue
    if title is None:
        continue
    if title == '':
        continue
    if title == 'Untitled':
        continue
    
    if 'NIST' in title:
        continue
    if 'HMNQ' in title:
        continue
    if '[' in title:
        continue
    if ']' in title:
        continue
    if 'h_' in title:
        continue
    if ('(' in title) and (')' not in title):
        continue
    if (')' in title) and ('(' not in title):
        continue
    if 'MMV' in title:
        continue
    if 'CHEBI' in title:
        continue
    if 'LQB' in title:
        continue
    if 'ot validated' in title:
        continue
    if 'isomer' in title:
        continue
    if '?' in title:
        continue
    if '//' in title:
        continue
    if 'Unknown' in title:
        continue
    if 'Spectral Match' in title:
        continue
    if 'Massbank' in title:
        continue
    if 'in silico' in title:
        continue
    if 'fragment' in title:
        continue

    try:
        m = Chem.MolFromSmiles(smi)
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = 'GNPS:{}'.format(gnps_index)
    except:
        continue
    
    if shortkey not in molShortkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
        gnps_index += 1


'''
print('19. Super Natural')
mols = Chem.SDMolSupplier('SuperNatural/sn3_database.sdf')
for m in tqdm(mols):
    if m is None:
        continue
    try:
        title = m.GetProp('parent_id')
        inchikey = inchi.MolToInchiKey(m)
        shortkey = inchikey[:14]
        molwt = AllChem.CalcExactMolWt(m)
        formula = AllChem.CalcMolFormula(m)
        smiles = Chem.MolToSmiles(m)
        dbid = m.GetProp('parent_id')
    except:
        continue
    
    if '*' in smiles:
        continue
    
    if inchikey not in molInChIkey:
        molTitle.append(title)
        molInChIkey.append(inchikey)
        molShortkey.append(shortkey)
        molExactmass.append(molwt)
        molFormula.append(formula)
        molSMILES.append(smiles)
        molDatabaseID.append([dbid])
    else:
        w = molInChIkey.index(inchikey)
        if dbid not in molDatabaseID[w]:
            molDatabaseID[w].append(dbid)
'''


print('finishing...')
molDatabaseID = [';;'.join(s) for s in tqdm(molDatabaseID)]

database = pd.DataFrame({'Title': molTitle,
                         'InChIkey': molInChIkey,
                         'Short InChIKey': molShortkey,
                         'Exact mass': molExactmass,
                         'Formula': molFormula,
                         'SMILES': molSMILES,
                         'Database IDs': molDatabaseID})

database.to_csv('database.csv', index = False)


print('predict class')
def predict_class(smi, timeout = 60):
    """
    Predict the class of a compound based on its SMILES representation using an external service.
    
    Parameters:
    - smi (str): The SMILES representation of the compound.
    - timeout (int): The timeout duration for the prediction request in seconds. Default is 60.
    
    Returns:
    - class_prediction (str or None): The predicted class of the compound, or None if prediction fails.
    """
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


database = pd.read_csv('database.csv')

from pycdk.pycdk import *
mols = [Chem.MolFromSmiles(s) for s in tqdm(database['SMILES'].values)]
writer = Chem.SDWriter('structures_1.sdf')
for mol in tqdm(mols):
    if mol is None:
        continue
    try:
        MolFromSmiles(Chem.MolToSmiles(mol))
    except:
        continue
    writer.write(mol)
writer.close()


Class, SuperClass = [], []
for i in tqdm(database.index):
    smi = database.loc[i,'SMILES']
    res = predict_class(smi)
    if res is None:
        Class.append('')
        SuperClass.append('')
    else:
        Class.append(res['class'])
        SuperClass.append(res['super_class'])
database['Class'] = Class
database['Super Class'] = SuperClass
database.to_csv('database.csv', index = False)


print('formula database')
alphabate = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'l', 'B', 'r', 'I', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
database = pd.read_csv('database.csv')
formula_database = database.loc[:, ['Formula']]
formula_database = formula_database.drop_duplicates()
formula, formula_mass = [], []
for f in tqdm(formula_database.iloc[:,0].values):
    if not all(char in alphabate for char in f):
        continue
    try:
        formula_mass.append(Formula(f).monoisotopic_mass)
        formula.append(f)
    except:
        continue

formula_database = pd.DataFrame({'Formula': formula, 'Exact mass': formula_mass})
formula_database = formula_database.sort_values('Exact mass')

formula_database.to_csv('formula_database.csv', index = False)


'''
from rdkit import Chem
from tqdm import tqdm

# Function to parse SDF and write to SMILES txt
def sdf_to_smiles(sdf_path, smiles_path):
    # Read the SDF file
    suppl = Chem.SDMolSupplier(sdf_path)
    
    # Open the SMILES txt file for writing
    with open(smiles_path, 'w') as smiles_file:
        for mol in tqdm(suppl):
            if mol is not None:  # Check if the molecule is valid
                smiles = Chem.MolToSmiles(mol)
                smiles_file.write(smiles + '\n')

# Specify the input SDF file path and output SMILES file path
sdf_path = 'database_sirius.sdf'
smiles_path = 'sirius_smiles.txt'

# Convert SDF to SMILES
sdf_to_smiles(sdf_path, smiles_path)
'''
