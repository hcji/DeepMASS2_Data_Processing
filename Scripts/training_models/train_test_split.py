# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 08:10:18 2022

@author: DELL
"""

import os
import numpy as np
import pickle
import random
from tqdm import tqdm
from matchms.importing import load_from_mgf

# CASMI
spectrums = [s for s in load_from_mgf('Example/CASMI/all_casmi.mgf')]
inchikey_casmi = list(set([s.get('inchikey')[:14] for s in spectrums]))


# NIST and GNPS
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums = pickle.load(file)
    
outfile = os.path.join(path_data, 'NIST2020/ALL_NIST20_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)
    
outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)

outfile = os.path.join(path_data, 'NIST2020/ALL_NIST20_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)

# In House
outfile = os.path.join(path_data, 'In_House/ALL_Inhouse_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)
    
outfile = os.path.join(path_data, 'In_House/ALL_Inhouse_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    spectrums += pickle.load(file)

inchikey_test = list(set(inchikey_casmi))


# Save
np.save('Saves/paper_version/inchikey_test.npy', inchikey_test)
