#!/home/neil.lu/.conda/envs/ringdown/bin/python
# coding: utf-8
# %%

import numpy as np
import os
import matplotlib.pyplot as pl
import glob
from matplotlib.offsetbox import AnchoredText

temp = np.array([])
files = glob.glob("./temp_results/220+221+330_realisation3_EC_*.dat")
    
for file in files:
    temp = np.append(temp, np.loadtxt(file))
#    os.remove('./'+str(file))
    
temp = temp.reshape((len(files),11))
np.savetxt('./results/220+221+330_realisation3_EC_ALL.dat', temp, header='start_time, SNRtot_MF, SNR220_MF, SNR221_MF, SNR330_MF, \
    nofilter, evidence220, evidence220_221, evidence220_221_330, np.log10(occams220_221), np.log10(occams220_221_330')

# %%
