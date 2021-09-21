#########################################################################################
# Load Modules
#########################################################################################

import numpy as np

import os

import sys

import astropy.units as u
from astropy.time import Time
from astropy.table import QTable

from numpy.lib.format import open_memmap
from baseband import vdif

from baseband_tasks.fourier import fft_maker
from baseband_tasks.dispersion import Dedisperse
from baseband_tasks.functions import Square
from baseband_tasks.shaping import ChangeSampleShape

fft_maker.set('numpy')

import time

from utils import DispersionMeasure, imshift
from StreamSearch_utils import *
from Analysis_tools import *

#########################################################################################
# Set up directories
#########################################################################################

codedir = '/home/serafinnadeau/Scripts/Chime_Crab_GP/chime_crab_gp/' 

# Input data directory
banddir = '/drives/scratch/ndisk_vdif_B0531+21_20210728/'
# Output data directory
testdir = '/drives/scratch/thierry/crab_archive/'
workdir = os.getcwd()

try:
    # Get date and time strings in format yyymmdd and hhmmss
    datestr = '20210728' #sys.argv[1]
    timestr = ''         #sys.argv[2]
except:
    raise Exception(f'sys.argv has length {len(sys.argv)}. datestr and timestr for dataset not set')

# Subdirectories for data output
istream = f'{testdir}{datestr}/istream/'
pulsedir = f'{testdir}{datestr}/pulsedir/'
splitdir = f'{testdir}{datestr}/splitdir/'
plotdir = f'{testdir}Plots/'

#########################################################################################
# Open 2D intensity stream memmap object
#########################################################################################

t0 = time.time()

os.chdir(istream)

# Get relevant stage 1 info from table
splittab = QTable.read(istream+'SplitTab.hdf5')

Nchan = splittab.meta['nchan']
Npol = splittab.meta['npol']
Nband = splittab.meta['subbands']
bw = int(Nchan / Nband)

binning = splittab.meta['binning']
startskip = splittab.meta['start_skipped']
endskip = splittab.meta['end_skipped']
Nfile = splittab.meta['vdif_files']
s_per_sample = splittab.meta['s_per_sample'] * binning

filecount = Nfile - (startskip + endskip)
samples_per_file = splittab.meta['samples_per_frame'] * splittab.meta['frames_per_file']

# Get shape of intensity stream
i_samples = int(filecount * samples_per_file / binning)
i_shape = (i_samples, bw * Nchan)

# Open intensity stream as memmap
I = open_memmap('i_stream.npy', dtype=np.float32, mode='r', 
                shape=i_shape)

print('Intensity stream loaded')
print(time.time() - t0, ' s')

#########################################################################################
# Open, dedisperse, collapse memmap stream chunk by chunk
#########################################################################################

t0 = time.time()

try:
    tab = QTable.read(istream+'search_tab.hdf5')
    dm = tab.meta['DM_guess']
    guess = False
    print(f'Previous run gives a DM guess of {dm}')
    
    stage='search_2'
    
except:
    dm = 56.74 # Stock DM for Crab
    guess = True
    print(f'No previous run: DM guess initially set to {dm}')
    
    stage='search_1'
    
def get_chunksize(I, maxchunk, dm):
    
    DM = DispersionMeasure(dm)
    dt = DM.time_delay(800*u.MHz, 400*u.MHz)
    time_waste = int(abs(dt.value / s_per_sample) + 1)
    
    W = 1
    timedim = I.shape[0]
    
    for w in range(2, maxchunk):
        if timedim % w == 0:
            if w > W:
                W = w
                
    return W + time_waste

w = get_chunksize(I, 50000, dm) # Set maxchunk as high as possible within memory constraints
    
stream1D = master(I, w=w, dm=dm, s_per_sample=s_per_sample)#, prometheus=[stage, promdir, registry, completion, timing])

print(f'Intensity stream master function run')
print(time.time() - t0, ' s')

#########################################################################################
# Correct 1D time stream for variations in background levels
#########################################################################################

t0 = time.time()

stream1D_corr = correct_stream(stream1D, istream, N=300)

print('')
print('Corrections applied')
print(time.time() - t0, ' s')

#############################################################################
# Search intensity stream for Giant Pulses
#############################################################################

t0 = time.time()

sigma = 2.5 # Set peak S/N cutoff
tab = streamsearch(stream1D_corr, splittab, sigma, banddir, istream, datestr, timestr, 
                   Nmax=False, Nmin=1024, dm=DispersionMeasure(dm), output=True)

print('')
print('Search completed')
print(time.time() - t0, ' s')

#############################################################################
# Estimate parameters for dataset
#############################################################################

t0 = time.time()

# Get spin frequency estimate
nu, _, _ = crab_nu(tab)

# Mark triggers as off-pulse (0), MP (1) or IP (2) 
## Compute phase of triggers based on estimated spin frequency
time_s = tab['off_s']
phase = time_s.value % (1/nu) * nu

component = np.zeros(len(tab))

## Find MP window triggers
h = np.histogram(phase, bins=np.linspace(0, 1, 101))
c, b = h[0], h[1]

i = np.argmax(c)

if i-2 >= 0:
    if i+3 < len(b):
        mask_mp = (phase < b[i-2]) + (phase > b[i+3])
    else:
        mask_mp = (phase > b[(i+3) % len(b)]) * (phase < b[i-2])
else:
    mask_mp = (phase < b[i-2]) * (phase > b[i+3])

component[~mask_mp] = 1

## Find IP window triggers
pcopy = phase[mask_mp]

h = np.histogram(pcopy, bins=np.linspace(0, 1, 101))
c, b = h[0], h[1]

i = np.argmax(c)

if i-2 >= 0:
    if i+3 < len(b):
        mask_ip = (phase < b[i-2]) + (phase > b[i+3])
    else:
        mask_ip = (phase > b[(i+3) % len(b)]) * (phase < b[i-2])
else:
    mask_ip = (phase < b[i-2]) * (phase > b[i+3])

component[~mask_ip] = 2
pcopy = phase[mask_ip * mask_mp]

## Update table with info
tab['phase'] = phase
tab['component'] = component
tab.meta['nu'] = nu
tab.meta['DM_guess'] = dm

tab.write(istream + f'search_tab.hdf5', path='search_info', overwrite=True)

print('')
print('Table updated')
print(time.time() - t0, ' s')