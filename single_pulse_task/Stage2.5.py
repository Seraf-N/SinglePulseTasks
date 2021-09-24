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

from s3_utils.py import *
from StreamSearch_utils import *

#########################################################################################
# Set config
#########################################################################################

prepulse = 2500 # Number of samples before pulse position
Nread = 15000 # Number of samples to be read in

#########################################################################################
# Set up directories
#########################################################################################

codedir = '/home/serafinnadeau/Scripts/Chime_Crab_GP/chime_crab_gp/'

banddir = '/drives/scratch/ndisk_vdif_B0531+21_20210728/'
testdir = '/drives/scratch/thierry/crab_archive/'#'/drives/CHF/8/crab_archive/'
workdir = os.getcwd()

datestr = '20210728'
timestr = ''

istream = f'{testdir}{datestr}/istream/'
pulsedir = f'{testdir}{datestr}/pulsedir/'
splitdir = f'{testdir}{datestr}/splitdir/'
plotdir = f'{testdir}Plots/'

#########################################################################################
# Open Stage 2 table and get pulse information
#########################################################################################

searchtab = QTable.read(istream + 'search_tab.hdf5')
splittab = QTable.read(istream + 'SplitTab.hdf5')

s_per_sample = searchtab.meta['s_per_sample']
dm = searchtab.meta['DM_guess']
Nband = searchtab.meta['subbands']
endskip = searchtab.meta['end_skipped']

x = np.argmax(searchtab['snr'])
pos = searchtab['pos'][x]

#########################################################################################
# Read in dedispersed pulse
#########################################################################################

dat = open_data(banddir, splittab, Nband, 0, endskip)
ded_dat = dedisperse_subbands(dat, tab, dm)
pulse = read_pulse(ded_dat, tab, pos=pos, prepulse=prepulse, Nread=Nread)

pulse_m = pulse.astype(np.float32)             # change to float32 to avoid overflows
pulse_m = np.sum(pulse_m**2, axis=-1)          # sum over squared polarisations
pulse_m = pulse_m / np.median(pulse_m, axis=0) # normalise the channels

for band in range(Nband):
    ded_dat[band].close()
    dat[band].close()

#########################################################################################
# Find DM which maximized peak S/N of pulse
#########################################################################################

t0 = time.time()

dms, sig = dm_fit(pulse_m, dm, Niter=4, L=100)
best_dm = dms[np.argmax(sig)].value

print('')
print('Best DM: ', best_dm)
print(time.time() - t0)

searchtab.meta['DM_guess'] = best_dm

searchtab.write(istream + f'search_tab.hdf5', path='search_info', overwrite=True)


