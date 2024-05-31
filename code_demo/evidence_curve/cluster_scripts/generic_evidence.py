#!/home/neil.lu/.conda/envs/ringdown/bin/python
# coding: utf-8
# %%

import numpy as np
from scipy.interpolate import interp1d
import qnm_filter
import sys
import qnm
import random
import argparse
import scipy.linalg as sl
from scipy.special import logsumexp
from pathlib import Path
import glob

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--filename")
argParser.add_argument("-m", "--modes")
args = argParser.parse_args()
index = int(args.filename)

# Determine the modes to calculate the threshold for
split_modes = args.modes.split(':')
modes_filt = split_modes[0].split("+")  # The overfiltered modes
modes_inj = split_modes[1].split("+")  # The injected modes
if not all(len(mode) == 3 for mode in modes_filt+modes_inj):
    raise Exception("Invalid mode, should be length 3 'lmn'")


def injection_evidence(start_time):
    fit = qnm_filter.Network(
        segment_length=0.2, srate=4096 * 1, t_init=start_time*mass)
    fit.original_data['H1'] = qnm_filter.RealData(signal, index=t_range)
    fit.detector_alignment()
    fit.pure_noise = {}
    fit.pure_noise['H1'] = qnm_filter.RealData(signalH_noise, index=t_range)
    fit.condition_data('original_data', remove_mean=False)
    fit.condition_data('pure_noise', remove_mean=False)
    fit.compute_acfs('pure_noise')
    fit.cholesky_decomposition()
    fit.first_index()

    _, evidence220 = qnm_filter.parallel_compute(
        fit, massspace, chispace, num_cpu=4, model_list=[(2, 2, 0, 'p')])
    likelihood220_221, evidence220_221 = qnm_filter.parallel_compute(fit, massspace, chispace, num_cpu=4,
                                                                     model_list=[(2, 2, 0, 'p'), (2, 2, 1, 'p')])
    likelihood220_221_330, evidence220_221_330 = qnm_filter.parallel_compute(fit, massspace, chispace, num_cpu=4,
                                                                             model_list=[(2, 2, 0, 'p'), (2, 2, 1, 'p'), (3, 3, 0, 'p')])
    contour220_221 = qnm_filter.find_credible_region(
        likelihood220_221, target_probability=0.95)
    contour220_221_330 = qnm_filter.find_credible_region(
        likelihood220_221_330, target_probability=0.95)
    occams220_221 = np.count_nonzero(
        likelihood220_221 > contour220_221)/len(mass_grid.flatten())
    occams220_221_330 = np.count_nonzero(
        likelihood220_221_330 > contour220_221_330)/len(mass_grid.flatten())
    nofilter = logsumexp(
        np.array(
            [fit.compute_likelihood(apply_filter=False)]
            * len(massspace)
            * len(chispace)
        )
    )

    fit.pure_nr = {}
    fit.pure_nr["H1"] = qnm_filter.RealData(
        signal220+signal221, index=t_range, ifo="H1")
    fit.condition_data('pure_nr', remove_mean=False)
    SNRtot_MF = fit.compute_SNR(
        fit.truncate_data(fit.original_data)["H1"],
        fit.truncate_data(fit.pure_nr)["H1"],
        "H1",
        False,
    )

    fit.pure_nr["H1"] = qnm_filter.RealData(signal220, index=t_range, ifo="H1")
    fit.original_data['H1'] = qnm_filter.RealData(
        signal220+signalH_noise, index=t_range, ifo="H1")
    fit.condition_data('pure_nr', remove_mean=False)
    fit.condition_data('original_data', remove_mean=False)
    SNR220_MF = fit.compute_SNR(
        fit.truncate_data(fit.original_data)["H1"],
        fit.truncate_data(fit.pure_nr)["H1"],
        "H1",
        False,
    )

    fit.pure_nr["H1"] = qnm_filter.RealData(signal221, index=t_range, ifo="H1")
    fit.original_data['H1'] = qnm_filter.RealData(
        signal221+signalH_noise, index=t_range, ifo="H1")
    fit.condition_data('pure_nr', remove_mean=False)
    fit.condition_data('original_data', remove_mean=False)
    SNR221_MF = fit.compute_SNR(
        fit.truncate_data(fit.original_data)["H1"],
        fit.truncate_data(fit.pure_nr)["H1"],
        "H1",
        False,
    )

    fit.pure_nr["H1"] = qnm_filter.RealData(signal330, index=t_range, ifo="H1")
    fit.original_data['H1'] = qnm_filter.RealData(
        signal330+signalH_noise, index=t_range, ifo="H1")
    fit.condition_data('pure_nr', remove_mean=False)
    fit.condition_data('original_data', remove_mean=False)
    SNR330_MF = fit.compute_SNR(
        fit.truncate_data(fit.original_data)["H1"],
        fit.truncate_data(fit.pure_nr)["H1"],
        "H1",
        False,
    )
    return np.array([start_time, SNRtot_MF, SNR220_MF, SNR221_MF, SNR330_MF,
                     nofilter, evidence220, evidence220_221, evidence220_221_330, np.log10(occams220_221), np.log10(occams220_221_330)])


# Parameters
mmax = 8.4 * 1e-21
delta_mass = 0.1
delta_chi = 0.008
massspace = np.arange(30, 160, delta_mass)
chispace = np.arange(0.0, 0.99, delta_chi)
mass_grid, chi_grid = np.meshgrid(massspace, chispace)

# Analysis
[t_range, signal, signal220, signal221, signal330, signalH_noise] = np.loadtxt(
    "./realisations/220+221+330_realisation3.dat")
mass = qnm_filter.Filter.mass_unit(100)  # Can be found in file header
time_space = np.arange(0, 50, 0.5)

result = injection_evidence(time_space[index])
np.savetxt(
    str(Path().absolute()) + "/temp_results/220+221+330_realisation3_EC_" +
    str(time_space[index]) + ".dat",
    result)
