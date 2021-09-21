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
from s3_utils import *

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

os.system(f'mkdir {testdir}')
os.system(f'mkdir {testdir}Plots')
os.system(f'mkdir {testdir}{datestr}')
os.system(f'mkdir {testdir}{datestr}/splitdir')
os.system(f'mkdir {testdir}{datestr}/istream')
os.system(f'mkdir {testdir}{datestr}/pulsedir')

#############################################################################
# Open tables and obtain relevant information
#############################################################################

splittab = QTable.read(istream + 'SplitTab.hdf5')
searchtab = QTable.read(istream + 'search_tab.hdf5')

Nband = searchtab.meta['subbands']
DM = searchtab.meta['DM_guess']

#############################################################################
# Select which pulses to archive based on S/N and max number of pulses
#############################################################################

def archive_lim(tab, percentile=95, cutoff=512):
    
    tab.sort('snr')
    tab = searchtab[::-1]
    
    m0 = tab['component'] == 0
    m1 = tab['component'] == 1
    m2 = tab['component'] == 2
    
    snrcut = np.percentile(tab['snr'][m0], percentile)
    
    snrmask = tab['snr'] > snrcut
    
    return tab[snrmask]

archivetab = archive_lim(searchtab)
archivetab.sort('off_s')

#############################################################################
# Open data and dedisperse to identified DM
#############################################################################

os.chdir(banddir)
dat = open_data(banddir, splittab, Nband)
os.chdir(workdir)

ded_dat = dedisperse_subbands(dat, searchtab, DM)

#############################################################################
# Save each dedispersed pulse in archivetab
#############################################################################

def archive_pulses(tab, dedispersed_data, prepulse, Nread, pulsedir, dtype=np.float16):
    
    t0 = time.time()
    
    snr = np.argsort(tab['snr'])
    snr_rank = np.arange(len(tab))[snr.argsort()]
    
    for i in range(len(tab)):
        
        pulse = read_pulse(dedispersed_data, tab, tab['pos'][i], prepulse, Nread)
        np.save(pulsedir + f'pulse_{snr_rank[i]:05d}.npy', pulse.astype(dtype))
        print(f'Pulse {i+1}/{len(tab)} dedispersed and written -- {(time.time() - t0)/60:.2f} minutes', end='                   \r')
        
        del pulse
    
    print('')
    
    return f'{len(tab)} pulses archived'

archive_pulses(archivetab, ded_dat, prepulse=2500, Nread=15000, pulsedir=pulsedir)

