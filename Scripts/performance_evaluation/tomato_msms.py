# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:52:49 2024

@author: DELL
"""


import matplotlib.pyplot as plt
from matchms.importing import load_from_mgf
from matchms.filtering import normalize_intensities
from matchms.filtering import remove_peaks_around_precursor_mz


def plot_comparision(ms1, ms2):
    mz1 = ms1.mz
    mz2 = ms2.mz
    intensity1 = ms1.intensities
    intensity2 = ms2.intensities
    
    plt.figure(figsize=(5, 3), dpi=300)
    plt.vlines(mz1, 0, intensity1, color = 'red')
    plt.vlines(mz2, 0, -intensity2, color = 'blue')
    plt.axhline(0, color = 'black')
    plt.xlabel('m/z', fontsize = 14)
    plt.ylabel('abundances', fontsize = 14)
    

mgf_1 = 'Example/Tomato/msms/Solasodine_TOM.mgf'
mgf_2 = 'Example/Tomato/msms/Solasodine_STD.mgf'
ms1 = normalize_intensities(remove_peaks_around_precursor_mz([s for s in load_from_mgf(mgf_1)][0]))
ms2 = normalize_intensities(remove_peaks_around_precursor_mz([s for s in load_from_mgf(mgf_2)][0]))
plot_comparision(ms1, ms2)


mgf_1 = 'Example/Tomato/msms/Tomatidine_TOM.mgf'
mgf_2 = 'Example/Tomato/msms/Tomatidine_STD.mgf'
ms1 = normalize_intensities(remove_peaks_around_precursor_mz([s for s in load_from_mgf(mgf_1)][0]))
ms2 = normalize_intensities(remove_peaks_around_precursor_mz([s for s in load_from_mgf(mgf_2)][0]))
plot_comparision(ms1, ms2)


mgf_1 = 'Example/Tomato/msms/Chlorogenic acid_TOM.mgf'
mgf_2 = 'Example/Tomato/msms/Chlorogenic acid_STD.mgf'
ms1 = normalize_intensities(remove_peaks_around_precursor_mz([s for s in load_from_mgf(mgf_1)][0]))
ms2 = normalize_intensities(remove_peaks_around_precursor_mz([s for s in load_from_mgf(mgf_2)][0]))
plot_comparision(ms1, ms2)


mgf_1 = 'Example/Tomato/msms/4-Hydroxycoumarin_TOM.mgf'
mgf_2 = 'Example/Tomato/msms/4-Hydroxycoumarin_STD.mgf'
ms1 = normalize_intensities(remove_peaks_around_precursor_mz([s for s in load_from_mgf(mgf_1)][0]))
ms2 = normalize_intensities(remove_peaks_around_precursor_mz([s for s in load_from_mgf(mgf_2)][0]))
plot_comparision(ms1, ms2)
