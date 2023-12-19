# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:49:51 2023

@author: DELL
"""


import os
import pickle
import numpy as np
from tqdm import tqdm
from matchms import Spectrum
from matchms.importing import load_from_msp
from matchms.filtering import default_filters
from matchms.filtering import add_parent_mass, derive_adduct_from_name

path_data = 'D:/DeepMASS2_Data_Processing/Datasets/MassBank'
filename = os.path.join(path_data, 'MassBank_NIST.msp')
spectrums = [s for s in tqdm(load_from_msp(filename))]

filename = os.path.join(path_data, 'MassBank_RIKEN.msp')
spectrums += [s for s in tqdm(load_from_msp(filename))]


def apply_filters(s):
    s = default_filters(s)
    s = derive_adduct_from_name(s)
    s = add_parent_mass(s, estimate_from_adduct=True)
    return s

spectrums = [apply_filters(s) for s in tqdm(spectrums) if s is not None]
np.save(os.path.join(path_data, 'preprocessed_spectrums.npy'), spectrums)


from matchms.filtering import harmonize_undefined_inchikey, harmonize_undefined_inchi, harmonize_undefined_smiles
from matchms.filtering import repair_inchi_inchikey_smiles

def clean_metadata(s):
    s = harmonize_undefined_inchikey(s)
    s = harmonize_undefined_inchi(s)
    s = harmonize_undefined_smiles(s)
    s = repair_inchi_inchikey_smiles(s)
    return s

spectrums = [clean_metadata(s) for s in tqdm(spectrums) if s is not None]
np.save(os.path.join(path_data, 'preprocessed_spectrums.npy'), spectrums)


from matchms.filtering import derive_inchi_from_smiles, derive_smiles_from_inchi
from matchms.filtering import derive_inchikey_from_inchi

def clean_metadata2(s):
    s = derive_inchi_from_smiles(s)
    s = derive_smiles_from_inchi(s)
    s = derive_inchikey_from_inchi(s)
    return s

spectrums = [clean_metadata2(s) for s in tqdm(spectrums) if s is not None]
np.save(os.path.join(path_data, 'preprocessed_spectrums.npy'), spectrums)


for spectrum in tqdm(spectrums):
    name_original = spectrum.get("compound_name")
    name = name_original.replace("F dial M", "")
    # Remove last word if likely not correct:
    if name.split(" ")[-1] in ["M", "M?", "?", "M+2H/2", "MS34+Na", "M]", "Cat+M]", "Unk", "--"]:
        name = " ".join(name.split(" ")[:-1]).strip()
    if name != name_original:
        print(f"Changed compound name from {name_original} to {name}.")
        spectrum.set("compound_name", name)
        

for spec in spectrums:
    if spec.get("adduct") in ['[M+CH3COO]-/[M-CH3]-',
                             '[M-H]-/[M-Ser]-',
                             '[M-CH3]-']:
        if spec.get("ionmode") != "negative":
            spec.set("ionmode", "negative")


from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz

def post_process(s):
    s = normalize_intensities(s)
    s = select_by_mz(s, mz_from=10.0, mz_to=1000)
    s = require_minimum_number_of_peaks(s, n_required=5)
    return s

spectrums = [post_process(s) for s in tqdm(spectrums)]
spectrums = [s for s in spectrums if s is not None]
np.save(os.path.join(path_data, 'preprocessed_spectrums.npy'), spectrums)

spectrums = np.load(os.path.join(path_data, 'preprocessed_spectrums.npy'), allow_pickle=True)
spectrums_positive = []
spectrums_negative = []
for i, s in enumerate(tqdm(spectrums)):
    try:
        new_s = Spectrum(mz = s.mz, intensities = s.intensities,
                         metadata = {'compound_name': s.get('compound_name'),
                                     'precursor_mz': s.get('precursor_mz'),
                                     'adduct': s.get('adduct'),
                                     'parent_mass': s.get('parent_mass'),
                                     'smiles': s.get('smiles'),
                                     'ionmode': s.get('ionmode'),
                                     'inchikey': s.get('inchikey'),
                                     'database': 'GNPS'})
    except:
        continue
    if new_s.get("ionmode") == "positive":
        spectrums_positive.append(new_s)
    elif new_s.get("ionmode") == "negative":
        spectrums_negative.append(new_s)
    else:
        pass

pickle.dump(spectrums_negative, 
            open(os.path.join(path_data, 'ALL_GNPS_220601_negative_cleaned.pickle'), "wb"))

pickle.dump(spectrums_positive, 
            open(os.path.join(path_data, 'ALL_GNPS_220601_positive_cleaned.pickle'), "wb"))