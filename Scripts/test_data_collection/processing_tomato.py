# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:27:35 2022

@author: DELL
"""


import pandas as pd

from matchms.exporting import save_as_mgf
from external.msdial import load_MS_DIAL_Alginment, remove_duplicate

pos_path = 'example/Tomato/tomato_positive_msdial.csv'
neg_path = 'example/Tomato/tomato_negative_msdial.csv'
pos_data = pd.read_csv(pos_path)
neg_data = pd.read_csv(neg_path)
pos_cols = list(pos_data.columns[32:])
neg_cols = list(neg_data.columns[32:])

spectrums_positive = load_MS_DIAL_Alginment('example/Tomato/tomato_positive_msdial.csv', sample_cols = pos_cols)
spectrums_negative = load_MS_DIAL_Alginment('example/Tomato/tomato_negative_msdial.csv', sample_cols = neg_cols)

spectrums_positive = [s for s in spectrums_positive if len(s.intensities[s.intensities > 0.05]) >= 3]
spectrums_positive = [s for s in spectrums_positive if s.get('parent_mass') is not None]
spectrums_negative = [s for s in spectrums_negative if len(s.intensities[s.intensities > 0.05]) >= 3]
spectrums_negative = [s for s in spectrums_negative if s.get('parent_mass') is not None]
spectrums_positive = [s.set('compound_name', 'Compound_{}'.format(i)) for i, s in enumerate(spectrums_positive)]
spectrums_negative = [s.set('compound_name', 'Compound_{}'.format(i)) for i, s in enumerate(spectrums_negative)]
save_as_mgf(spectrums_positive, 'example/Tomato/ms_ms_tomato_all_positive.mgf')
save_as_mgf(spectrums_negative, 'example/Tomato/ms_ms_tomato_all_negative.mgf')

spectrums = spectrums_positive + spectrums_negative
spectrums = remove_duplicate(spectrums)

save_as_mgf(spectrums, 'example/Tomato/ms_ms_tomato_identified.mgf')
