# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:05:49 2024

@author: DELL
"""


import os
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, inchi
from pyteomics import mzml
from matchms import Spectrum
from matchms import filtering as msfilters
from matchms.importing import load_from_mzml, load_from_mgf
from matchms.exporting import save_as_mgf, save_as_msp
from matchms.importing.parsing_utils import parse_mzml_mzxml_metadata

def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = msfilters.default_filters(s)
    s = msfilters.add_parent_mass(s)
    s = msfilters.normalize_intensities(s)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=2000)
    return s


files = os.listdir('Example/Anticancer/raw_mzml')
compounds = pd.read_excel('Example/Anticancer/CompList.xlsx')

challenge_ms = []
for group in tqdm(range(1, 6)):    
    spectrums_ = [s for s in mzml.read('Example/Anticancer/raw_mzml/anticancer-{}.mzML'.format(group))]
    spectrums_ = [s for s in spectrums_ if s['ms level'] == 2]
    spectrums_metadata = [parse_mzml_mzxml_metadata(s) for s in spectrums_]
    precursor_mzs_ = np.array([s['precursor_mz'] for s in spectrums_metadata])
    retention_times_ = np.array([s['scan_start_time'][0] for s in spectrums_metadata])
    total_intensities_ = np.array([s['total ion current'] for s in spectrums_])
    compounds_ = compounds.loc[compounds['Group'] == group, :]
    for i in compounds_.index:
        name = compounds_.loc[i, 'ProductName']
        smi = compounds_.loc[i, 'Smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        inchikey = inchi.MolToInchiKey(mol)
        precursor_type = '[M+H]+'
        exact_mass = AllChem.CalcExactMolWt(mol)
        precursor_mz = 1.007276 + exact_mass
        formula = AllChem.CalcMolFormula(mol)
        ionmode = 'positive'
        
        wh = np.where(np.abs(precursor_mzs_ - precursor_mz) < 0.01)[0]
        if len(wh) < 2:
            continue
        wh = wh[np.argmax(total_intensities_[wh])]
        mz = spectrums_[wh]['m/z array']
        intensities = spectrums_[wh]['intensity array']
        
        s_new = Spectrum(mz = mz, intensities = intensities, 
                         metadata = {'name': 'challenge_{}'.format(i),
                                     'precursor_mz': precursor_mz})
        s_new.set('formula', formula)
        s_new.set('smiles', smi)
        s_new.set('inchikey', inchikey)
        s_new.set('adduct', precursor_type)
        s_new = spectrum_processing(s_new)
        challenge_ms.append(s_new)

# save as mgf
save_as_mgf(list(challenge_ms), 'Example/Anticancer/export.mgf')

challenge_ms = [s for s in load_from_mgf('Example/Anticancer/export.mgf')]
for i, s in enumerate(challenge_ms):
    j = s.get('compound_name').split('_')[-1]
    path = 'Example/Anticancer/msp/challenge_{}.msp'.format(j)
    save_as_msp([challenge_ms[i]], path)
    
    with open(path) as msp:
        lines = msp.readlines()
        lines = [l.replace('_', '') for l in lines]
        lines = [l.replace('ADDUCT', 'PRECURSORTYPE') for l in lines]
    with open(path, 'w') as msp:
        msp.writelines(lines)

    # only for ms-finder
    path_msfinder = path.replace('/msp/', '/msfinder/')
    
    if float(s.metadata['parent_mass']) >= 1000:
        continue        
    with open(path_msfinder, 'w') as msp:
        msp.writelines(lines)

