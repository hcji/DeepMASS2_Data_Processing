# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:30:51 2023

@author: DELL
"""


import os
import pickle
import numpy as np
from tqdm import tqdm

# positive
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_positive_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference = pickle.load(file)

for s in tqdm(reference):
    if s.get('precursor_mz') is None:
        s.set('precursor_mz', 0)

pmz = np.array([s.get('precursor_mz') for s in tqdm(reference)])
reference = np.array(reference)[np.argsort(pmz)]

pickle.dump(reference, 
            open(os.path.join('Saves/public_version/references_spectrums_positive.pickle'), "wb"))


# negative
path_data = 'D:/DeepMASS2_Data_Processing/Datasets'

outfile = os.path.join(path_data, 'GNPS_all/ALL_GNPS_220601_negative_cleaned.pickle')
with open(outfile, 'rb') as file:
    reference = pickle.load(file)

for s in tqdm(reference):
    if s.get('precursor_mz') is None:
        s.set('precursor_mz', 0)

pmz = np.array([s.get('precursor_mz') for s in tqdm(reference)])
reference = np.array(reference)[np.argsort(pmz)]

pickle.dump(reference, 
            open(os.path.join('Saves/public_version/references_spectrums_negative.pickle'), "wb"))
