#!/usr/bin/env python
# coding: utf-8
# %%


import numpy as np
from scipy.interpolate import interp1d
import qnm_filter
import sys
import qnm
import random
import argparse
import time
tic = time.time()

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--filename")

args = argParser.parse_args()

print("%s" % args.filename)


sampling_frequency = 4096 * 1  # in Hz
duration = 8  # in second

mass_in_solar = random.uniform(45, 150)
chi_inject = random.uniform(0.0, 0.85)
mass = qnm_filter.Filter.mass_unit(mass_in_solar)  # in solar mass
omega220, _, _ = qnm.modes_cache(s=-2, l=2, m=2, n=0)(a=chi_inject)
omega330, _, _ = qnm.modes_cache(s=-2, l=3, m=3, n=0)(a=chi_inject)
model_list = [(2, 2, 0, "p"), (2, 0, 0, "p")]

credibility220330_list = []
snr_list = []
for i in range(200):
    print(i)
    t_range = np.arange(-duration / 2, duration / 2, 1 / sampling_frequency)

    mmax = 8.4 * 1e-21
    mmin = 1.071428571428571e-22

    A220x = random.uniform(-mmax, mmax)
    A220y = random.uniform(-mmax, mmax)

    A330x = random.uniform(-mmax, mmax)
    A330y = random.uniform(-mmax, mmax)

    signal = t_range * 0

    signal[t_range > 0] = np.real(
        np.exp(-1j * omega220 * np.abs(t_range[t_range > 0] / mass))
        * (A220x + 1j * A220y)
        + np.exp(-1j * omega330 * np.abs(t_range[t_range > 0] / mass))
        * (A330x + 1j * A330y)
    )

    bilby_ifo = qnm_filter.set_bilby_predefined_ifo(
        "H1", sampling_frequency, duration, start_time=-duration / 2
    )

    signalH_noise = qnm_filter.bilby_get_strain(bilby_ifo, 0.0)


    signalH_no_noise = qnm_filter.RealData(signal, index=t_range, ifo="H1")
    signalH = signalH_no_noise + signalH_noise

    fit = qnm_filter.Network(segment_length=0.2, srate=4096 * 1, t_init=3.0 * mass)

    fit.original_data["H1"] = signalH
    fit.detector_alignment()

    fit.pure_noise = {}
    fit.pure_noise["H1"] = signalH_noise

    fit.pure_nr = {}
    fit.pure_nr["H1"] = signalH_no_noise

    fit.condition_data("original_data")
    fit.condition_data("pure_noise")
    fit.condition_data("pure_nr")
    fit.compute_acfs("pure_noise")

    fit.cholesky_decomposition()

    delta_chi = 0.01
    delta_mass = 0.029197 * mass_in_solar
    massspace = np.arange(45, 150, 0.5)
    chispace = np.arange(0.0, 0.85, delta_chi)
    mass_grid, chi_grid = np.meshgrid(massspace, chispace)

    fit.first_index()
    likelihood_data220330, _ = qnm_filter.parallel_compute_cached_omega(
        fit,
        massspace,
        chispace,
        num_cpu=-12,
        model_list=model_list,
    )

    snr = fit.compute_SNR(
        fit.truncate_data(fit.original_data)["H1"],
        fit.truncate_data(fit.pure_nr)["H1"],
        "H1",
        False,
    )

    credibility220330 = qnm_filter.credibility_of_mass_spin(
        likelihood_data220330,
        fit,
        mass_in_solar,
        chi_inject,
        model_list=model_list,
    )

    credibility220330_list.append(credibility220330)
    snr_list.append(snr)

np.savetxt(
    "results/credibility_toy_model_220+330" + args.filename + ".dat",
    np.c_[
        snr_list, credibility220330_list
    ],
)

print((time.time() - tic)/60)