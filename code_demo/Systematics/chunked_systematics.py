#!/nlu/RD/bin/python3
import matplotlib.pyplot as pl
import numpy as np
import qnm_filter
from gwpy.timeseries import TimeSeries
import copy
import time
import sys

tic = time.time()
arg = int(sys.argv[1])
home_dir = "/home/nlu/no_noise_inj/"
filters = [(2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 1, 0), (3, 3, 0), (4, 4, 0)]

H_filename = "H-H1_NR_INJECTED-1126259448-16_TEOBResumS_GR_q_0.8_chi1_0.1_chi2_0.1_M72_dist452_incl1p59_ra1p68_decm1p27_psi3p93_flow7_nonoise_aligned.gwf"
L_filename = "L-L1_NR_INJECTED-1126259448-16_TEOBResumS_GR_q_0.8_chi1_0.1_chi2_0.1_M72_dist452_incl1p59_ra1p68_decm1p27_psi3p93_flow7_nonoise_aligned.gwf"

H_data = TimeSeries.read(home_dir + H_filename, 'H1:NR_INJECTED')
L_data = TimeSeries.read(home_dir + L_filename, 'L1:NR_INJECTED')

H_waveform = qnm_filter.Data(H_data.value, index=H_data.times.value)
L_waveform = qnm_filter.Data(L_data.value, index=L_data.times.value)
peak_time = H_data.times.value[np.argmax(H_waveform)]
SSB_peak_time = peak_time - 0.014685396838313368

input = dict(model_list=[(2, 2, 0)],  # l, m, n
             # trucation time (geocenter, in second)
             t_init=SSB_peak_time+0e-3,  # Calculated from SNR+t_init notebook
             # length of the analysis window (in second)
             window_width=0.2,
             # sampling rate after conditioning (in Hz)
             srate=2048,
             # sky localization
             ra=1.95, dec=-1.27,
             # lower limit of the high-pass filter (in Hz)
             flow=20)

delta_mass = 0.2
delta_chi = 0.005
massspace = np.arange(34, 140, delta_mass)
chispace = np.arange(0.0, 0.95, delta_chi)
mass_grid, chi_grid = np.meshgrid(massspace, chispace)

fit = qnm_filter.Network(**input)
fit.original_data['H1'] = H_waveform
fit.detector_alignment()
fit.condition_data('original_data', **input, trim=0.0)
fit.compute_acfs('original_data')
temp_acf = np.full(input['srate'], 0, dtype=np.double)
temp_acf[0] = 1e-23**2
fit.acfs['H1'] = qnm_filter.Data(temp_acf, index=fit.acfs['H1'].index)
fit.cholesky_decomposition()
fit.first_index()

index_spacing = 1
num_iteration = 1
initial_offset = -10+num_iteration*arg*index_spacing
t_array, saved_log_evidence, average_values, MAP_values = qnm_filter.evidence_parallel(fit, index_spacing,
                                                                                       num_iteration,
                                                                                       initial_offset, massspace, chispace, num_cpu=1,
                                                                                       verbosity=False, model_list=input['model_list'])

saved_log_evidence /= np.log(10)  # ln to lg
np.savetxt(home_dir+"results/220/chunk" +
           str(arg) + ".txt", np.c_[t_array, saved_log_evidence, average_values, MAP_values])


toc = time.time()
print((toc-tic)/60)
