# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:52:49 2024

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
from matchms.importing import load_from_mgf
from matchms.filtering import normalize_intensities, reduce_to_number_of_peaks
from matchms.filtering import remove_peaks_around_precursor_mz


def reduce_peaks(mzs, intensities, slice_width):
    from collections import defaultdict
    slices = defaultdict(list)
    for i in range(len(mzs)):
        mz, intensity = mzs[i], intensities[i]
        slice_key = int(mz // slice_width)
        slices[slice_key].append((mz, intensity))

    # Find the highest peak in each slice
    highest_peaks = []
    for peaks in slices.values():
        highest_peak = max(peaks, key=lambda x: x[1])
        highest_peaks.append(highest_peak)
    highest_peaks = np.array(highest_peaks)
    return highest_peaks[:,0], highest_peaks[:,1]


def plot_comparision(ms1, ms2):
    mz1 = ms1.mz
    mz2 = ms2.mz
    intensity1 = ms1.intensities
    intensity2 = ms2.intensities
    
    mz1, intensity1 = reduce_peaks(mz1, intensity1, 3)
    mz2, intensity2 = reduce_peaks(mz2, intensity2, 3)
    
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
ms1 = normalize_intensities([s for s in load_from_mgf(mgf_1)][0])
ms2 = normalize_intensities([s for s in load_from_mgf(mgf_2)][0])
plot_comparision(ms1, ms2)
