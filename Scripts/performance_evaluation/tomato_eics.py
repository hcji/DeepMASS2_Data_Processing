# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:52:49 2024

@author: DELL
"""


import pymzml
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_eic(mzml_file, target_mz, rt_min, rt_max, mz_tolerance=0.01):
    """Extract EIC for a given target m/z and retention time range from an mzML file.

    Args:
        mzml_file (str): Path to the mzML file.
        target_mz (float): Target m/z value.
        rt_min (float): Minimum retention time in minutes.
        rt_max (float): Maximum retention time in minutes.
        mz_tolerance (float): Tolerance for m/z matching.

    Returns:
        tuple: Retention times and corresponding intensities.
    """
    run = pymzml.run.Reader(mzml_file)
    rts = []
    intensities = []

    for spectrum in run:
        if spectrum['ms level'] == 1:
            rt = spectrum.scan_time_in_minutes() * 60
            if rt_min <= rt <= rt_max:
                intensity = 0
                for mz, i in spectrum.peaks('centroided'):
                    if abs(mz - target_mz) <= mz_tolerance:
                        intensity += i
                rts.append(rt)
                intensities.append(intensity)
    
    return rts, intensities


def calculate_row_means(matrix, group_size=3):
    """
    Calculate the mean values of a matrix such that each row of the output matrix
    represents the mean values of every 'group_size' rows of the input matrix.

    Args:
        matrix (np.ndarray): Input matrix.
        group_size (int): Number of rows to group together for calculating the mean.

    Returns:
        np.ndarray: Output matrix with mean values.
    """
    # Reshape the matrix to group rows
    reshaped_matrix = matrix.reshape(-1, group_size, matrix.shape[1])
    # Calculate the mean along the second axis (rows within each group)
    mean_matrix = reshaped_matrix.mean(axis=1)
    return mean_matrix


def plot_eic(rts, mean_intensities):
    plt.figure(figsize=(4, 2), dpi=300)
    plt.plot(rts, mean_intensities[0,:], label = 'MG', color = '#4D0C7B')
    plt.plot(rts, mean_intensities[1,:], label = 'Br3', color = '#0C7B2D')
    plt.plot(rts, mean_intensities[2,:], label = 'Br7', color = '#11A9C4')
    plt.plot(rts, mean_intensities[3,:], label = 'Br10', color = '#0D4584')
    plt.plot(rts, mean_intensities[4,:], label = 'Br15', color = '#8E0E0E')
    plt.xlabel("Retention Time (s)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()



data_files_pos = ["E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-10.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-11.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-40.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-59.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-80.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-84.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-99.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-118.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-123.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-125.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-132.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-133.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-183.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-194.mzML",
                  "E:/Data/tomato_data/Positive/MH02-009-2-1ul-pos-201.mzML"
                  ]

data_files_neg = ["E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-010.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-011.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-040.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-059.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-080.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-084.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-099.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-118.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-123.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-125.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-132.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-133.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-183.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-194.mzML",
                  "E:/Data/tomato_data/Negative/MH2102-009-2ul-neg-201.mzML"
                  ]


# Solasodine
target_mz = 414.3354
mz_tolerance = 0.05
rt_min, rt_max = 780, 805
mzml_files = data_files_pos

eic_rts = np.linspace(rt_min, rt_max, 100)
eic_intensities = []
for i, mzml_file in enumerate(tqdm(mzml_files)):
    if os.path.exists(mzml_file):
        rts, intensities = extract_eic(mzml_file, target_mz, rt_min, rt_max, mz_tolerance)
        intensities = np.interp(eic_rts, rts, intensities)
        eic_intensities.append(intensities)

eic_intensities = np.array(eic_intensities)
mean_intensities = calculate_row_means(eic_intensities, group_size=3)
plot_eic(eic_rts, mean_intensities)


# Tomatidine
target_mz = 416.3523
mz_tolerance = 0.05
rt_min, rt_max = 785, 810
mzml_files = data_files_pos

eic_rts = np.linspace(rt_min, rt_max, 100)
eic_intensities = []
for i, mzml_file in enumerate(tqdm(mzml_files)):
    if os.path.exists(mzml_file):
        rts, intensities = extract_eic(mzml_file, target_mz, rt_min, rt_max, mz_tolerance)
        intensities = np.interp(eic_rts, rts, intensities)
        eic_intensities.append(intensities)

eic_intensities = np.array(eic_intensities)
mean_intensities = calculate_row_means(eic_intensities, group_size=3)
plot_eic(eic_rts, mean_intensities)


# Cryptochlorogenic acid-like
target_mz = 353.085
mz_tolerance = 0.05
rt_min, rt_max = 395, 425
mzml_files = data_files_neg

eic_rts = np.linspace(rt_min, rt_max, 100)
eic_intensities = []
for i, mzml_file in enumerate(tqdm(mzml_files)):
    if os.path.exists(mzml_file):
        rts, intensities = extract_eic(mzml_file, target_mz, rt_min, rt_max, mz_tolerance)
        intensities = np.interp(eic_rts, rts, intensities)
        eic_intensities.append(intensities)

eic_intensities = np.array(eic_intensities)
mean_intensities = calculate_row_means(eic_intensities, group_size=3)
plot_eic(eic_rts, mean_intensities)


# Hydroxycoumarin-like
target_mz = 163.039
mz_tolerance = 0.01
rt_min, rt_max = 474, 494
mzml_files = data_files_pos

eic_rts = np.linspace(rt_min, rt_max, 100)
eic_intensities = []
for i, mzml_file in enumerate(tqdm(mzml_files)):
    if os.path.exists(mzml_file):
        rts, intensities = extract_eic(mzml_file, target_mz, rt_min, rt_max, mz_tolerance)
        intensities = np.interp(eic_rts, rts, intensities)
        eic_intensities.append(intensities)

eic_intensities = np.array(eic_intensities)
mean_intensities = calculate_row_means(eic_intensities, group_size=3)
plot_eic(eic_rts, mean_intensities)
