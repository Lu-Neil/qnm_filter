#!/home/neil.lu/.conda/envs/ringdown/bin/python
# coding: utf-8
# %%


import numpy as np
import qnm_filter
import qnm
import random
import argparse
from scipy.special import logsumexp
from pathlib import Path
import glob
from gwpy.timeseries import TimeSeries

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--filename")
argParser.add_argument("-m", "--modes")
args = argParser.parse_args()
index = int(args.filename)

# Determine the modes to calculate the threshold for
split_modes = args.modes.split(':')
modes_filt = split_modes[0].split("+") #The overfiltered modes
modes_inj = split_modes[1].split("+") #The injected modes
if not all(len(mode)==3 for mode in modes_filt+modes_inj):
    raise Exception("Invalid mode, should be length 3 'lmn'")

duration = 4
fsamp = 4096

def signal_creator(noise_scale):
    global signal, signal_inj, mass, t_range, signalH_noise
    # Remnant properties
    mass_in_solar = random.uniform(50, 120)
    chi_inject = random.uniform(0.0, 0.95)
    injected = (mass_in_solar, chi_inject)
    mass = qnm_filter.Filter.mass_unit(injected[0]) #unit of time
    
    # Complex frequency of the modes
    omega_dict = {}
    for mode in modes_inj:
    	omega_dict[mode] = qnm.modes_cache(s=-2,l=int(mode[0]),m=int(mode[1]),n=int(mode[2]))(a=injected[1])[0]

    amp = 8.4 * 1e-21
    t_range = np.arange(-duration/2, duration/2, 1/fsamp)
    
    # Creating the signal
    signal_inj = np.zeros(len(t_range))
    for keys, omega in omega_dict.items():
        phase = random.uniform(0, 2*np.pi)
        temp_signal = np.real(amp * np.exp(1j*phase) * np.exp(-1j * omega * np.abs(t_range / mass)))
        signal_inj = np.add(signal_inj, temp_signal)
    
    bilby_ifo = qnm_filter.set_bilby_predefined_ifo(
    "H1", fsamp, duration, start_time=-duration / 2)
    signalH_noise = noise_scale*qnm_filter.bilby_get_strain(bilby_ifo, 0.0)
    signal = signal_inj + signalH_noise

def injection_evidence():
    fit = qnm_filter.Network(segment_length=0.2, srate=4096, t_init=3.0*mass)
    fit.original_data['H1'] = qnm_filter.RealData(signal, index=t_range)
    fit.detector_alignment()
    fit.pure_noise = {}
    fit.pure_noise['H1'] = qnm_filter.RealData(signalH_noise, index=t_range)
    fit.condition_data('original_data', trim=False, remove_mean=False)
    fit.condition_data('pure_noise', trim=False, remove_mean=False)
    fit.compute_acfs('original_data')
    fit.cholesky_decomposition()
    fit.first_index()
    model_list_inj = [(int(m[0]),int(m[1]),int(m[2]),'p') for m in modes_inj]
    model_list_filt = [(int(m[0]),int(m[1]),int(m[2]),'p') for m in modes_filt]
    
    _, evidence_inj = qnm_filter.parallel_compute(fit, massspace, chispace, num_cpu = 4,
                                                 model_list=model_list_inj)
    _, evidence_filt = qnm_filter.parallel_compute(fit, massspace, chispace, num_cpu = 4, 
                                                 model_list=model_list_filt)
    nofilter = logsumexp(
        np.array(
            [fit.compute_likelihood(apply_filter=False)]
            * len(massspace)
            * len(chispace)
        )
    )
    
    fit.pure_nr = {}
    fit.pure_nr["H1"] = qnm_filter.RealData(signal_inj, index=t_range, ifo="H1")
    fit.condition_data('pure_nr', trim=False, remove_mean=False)
    SNRtot_MF = fit.compute_SNR(
        fit.truncate_data(fit.original_data)["H1"],
        fit.truncate_data(fit.pure_nr)["H1"],
        "H1",
        False,
    )
    return np.array([SNRtot_MF, evidence_inj, evidence_filt, nofilter])

# Parameters
delta_mass = 0.1
delta_chi = 0.008
massspace = np.arange(10, 160, delta_mass)
chispace = np.arange(0.0, 0.99, delta_chi)
mass_grid, chi_grid = np.meshgrid(massspace, chispace)

# Analysis
noise_scale = np.power(random.uniform(0.3, 3), -1)
signal_creator(noise_scale)
result = injection_evidence()

np.savetxt(
    str(Path().absolute()) + "/temp_results/O4_design_"+args.modes+"_"+str(index) + ".dat",
    result,
)
# %%
