#!/home/neil.lu/.conda/envs/ringdown/bin/python
# coding: utf-8
# %%

import numpy as np
import os
import glob
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--modes")
args = argParser.parse_args()

temp = np.array([])
files = glob.glob("./temp_results/O4_design_"+args.modes+"*.dat")
    
for file in files:
    temp = np.append(temp, np.loadtxt(file))
#    os.remove(os.fsencode(results_str+filename))
    
temp = temp.reshape((len(files),4))
[SNRtot_MF, evidence_inj, evidence_filt, nofilter] = temp.T
thresh = np.median(evidence_filt - evidence_inj)+3*np.std(evidence_filt - evidence_inj)
np.savetxt('results/O4_design_overfiltered_'+args.modes+'_ALL.dat', temp, header="thresh=%.2f, format=SNRtot_MF, evidence_inj, evidence_filt, nofilter]" % thresh)


#%%
