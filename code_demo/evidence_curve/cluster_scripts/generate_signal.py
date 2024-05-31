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

index = len(glob.glob('./realisations/*'))

def signal_creator(noise_scale):
    global signal, signal220, signal221, signal330, t_range, signalH_noise, mass
    injected = (100, 0.4)
    mass = qnm_filter.Filter.mass_unit(injected[0])
    omega220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)(a=injected[1])[0]
    omega221 = qnm.modes_cache(s=-2,l=2,m=2,n=1)(a=injected[1])[0]
    omega330 = qnm.modes_cache(s=-2,l=3,m=3,n=0)(a=injected[1])[0]

    mmax = 8.4 * 1e-21 
    phase1 = random.uniform(0, 2*np.pi)
    A220x = mmax*np.cos(phase1)
    A220y = mmax*np.sin(phase1)
    phase2 = random.uniform(0, 2*np.pi)
    A221x = mmax*np.cos(phase2)
    A221y = mmax*np.sin(phase2)
    phase3 = random.uniform(0, 2*np.pi)
    A330x = mmax*np.cos(phase2)
    A330y = mmax*np.sin(phase2)

    amp220 = random.uniform(0.5, 1)
    amp221 = random.uniform(0.1, 0.5)
    amp330 = random.uniform(0.4, 0.8)

    sampling_frequency = 4096 * 1  # in Hz
    duration = 2  # in second
    t_range = np.arange(-duration / 2, duration / 2, 1 / sampling_frequency)
    signal220 = np.real(amp220 * (A220x + 1j * A220y) * np.exp(-1j * omega220 * np.abs(t_range / mass)))
    signal221 = np.real(amp221 * (A221x + 1j * A221y) * np.exp(-1j * omega221 * np.abs(t_range / mass)))
    signal330 = np.real(amp330 * (A330x + 1j * A330y) * np.exp(-1j * omega330 * np.abs(t_range / mass)))

    bilby_ifo = qnm_filter.set_bilby_predefined_ifo(
    "H1", sampling_frequency, duration, start_time=-duration / 2
)
    signalH_noise = noise_scale*qnm_filter.bilby_get_strain(bilby_ifo, 0.0)
    signal = signal220+signal221+signal330+signalH_noise


# Analysis
noise_scale = 0.2**(-1) #np.power(random.uniform(0.3, 3), -1)
signal_creator(noise_scale)

np.savetxt(
    str(Path().absolute()) + "/realisations/220+221+330_realisation" + str(index) + ".dat",
    [t_range, signal, signal220, signal221, signal330, signalH_noise], header='Mass=%.3f' % mass)
