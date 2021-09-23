##########################################################################
# Import modules
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

import os

import astropy.units as u
from astropy.table import QTable

import sys

from scipy.stats import binned_statistic as bs

import time
from functools import reduce

from numpy.lib.format import open_memmap
from baseband_tasks.functions import Square
from baseband import vdif

from utils import imbin

##########################################################################
# Set up relevant directories
##########################################################################

import sys
try:
    datestr = sys.argv[1]#'20190626'
    timestr = sys.argv[2]#'191438'
except:
    #raise Exception(f'sys.argv has length {len(sys.argv)}. datestr and timestr for dataset not set')
    datestr = '20210728'
    timestr = ''

codedir = '/home/serafinnadeau/Scripts/Chime_Crab_GP/chime_crab_gp/'
workdir = os.getcwd()

banddir = '/drives/scratch/ndisk_vdif_B0531+21_20210728/'
testdir = '/drives/scratch/thierry/crab_archive/'#'/drives/CHF/8/crab_archive/'

istream = f'{testdir}{datestr}/istream/'
pulsedir = f'{testdir}{datestr}/pulsedir'
splitdir = f'{testdir}{datestr}/splitdir'
plotdir = f'{testdir}Plots'

os.system(f'mkdir {testdir}')
os.system(f'mkdir {testdir}Plots')
os.system(f'mkdir {testdir}{datestr}')
os.system(f'mkdir {testdir}{datestr}/splitdir')
os.system(f'mkdir {testdir}{datestr}/istream')
os.system(f'mkdir {testdir}{datestr}/pulsedir')

os.system(f'rm {istream}i_stream.npy')
os.system(f'rm {istream}SplitTab.hdf5')

##########################################################################
# Obtain data file paths
##########################################################################

files = os.listdir(banddir)

fpath = []
fnum = []
fband = []
fname = []

for file in files:
    if file.endswith('.vdif'):
        fn, fb = file.split('.')[0].split('_')
        fnum += [fn]
        fband += [fb]
        fname += [file]
        fpath += [banddir + file]

ftab = QTable([fname, fnum, fband, fpath],
              names = ['fname', 'fnumber', 'fband', 'fpath'],
              meta = {'samples_per_file': 50000,
                      'samples_per_frame': 625,
                      'frames_per_file': 80,
                      'binning': 100,
                      's_per_sample': 2.56e-6,
                      'nchan': 1024,
                      'npol': 2,
                      'pol_type': 'complex',
                      'file_path': banddir})

ftab.sort(['fnumber', 'fband'])

##########################################################################
# Get number of files and subbands
##########################################################################

Nfile = np.max(ftab['fnumber'].astype(int)) + 1
Nband = np.max(ftab['fband'].astype(int)) + 1

ftab.meta['subbands'] = Nband
ftab.meta['vdif_files'] = Nfile

##########################################################################
# Open data streams
##########################################################################

bands = []
data = []
startskip = 157 # how many files are skipped at start of data
endskip = 1 # how many files are skipped at end of data

ftab.meta['start_skipped'] = startskip
ftab.meta['end_skipped'] = endskip

for i in range(Nband):
    mask = ftab['fband'].astype(int) == i
    files = ftab[mask]['fpath']
    data += [vdif.open(files[startskip:-endskip], 'rs', sample_rate=1/(ftab.meta['s_per_sample']*u.s), verify=False)]
    bands += [Square(data[-1])]

ftab.remove_column('fpath')

ftab.meta['start_time'] = data[0].start_time.isot
ftab.meta['stop_time'] = data[0].stop_time.isot

###########################################################################
# Set up memmap for saving binned intensity stream
###########################################################################

t0 = time.time()

Nchan = ftab.meta['nchan']
Npol = ftab.meta['npol']
bw = int(Nchan / Nband)

binning = ftab.meta['binning']

filecount = Nfile - (startskip + endskip)
samples_per_file = ftab.meta['samples_per_frame'] * ftab.meta['frames_per_file']

i_samples = int(filecount * samples_per_file / binning)
i_shape = (i_samples, Nchan)

i_stream = open_memmap(istream + 'i_stream.npy', dtype=np.float16,
                       mode='w+', shape=i_shape)

print(f'Intensity stream memmap initialized')
print(time.time() - t0, ' s')

##########################################################################
# Compute how many files of data to read in at a time:
#     - ensure that fileskip * samples_per_file is a multiple of 100
#     - ensure that filecount is a multiple of fileskip
#     - choose the highest valid integer between 1 and 10
##########################################################################

def get_fileskip(filecount, samples_per_file):

    skips = []

    maxskip = int(500000 / samples_per_file)
    if maxskip <= 1:
        maxskip = 1

    for fskip in range(1, maxskip+1):
        if (fskip * samples_per_file) % 100 == 0:
            if filecount % fskip == 0:
                skips += [fskip]

    return int(np.max(skips))

fileskip = get_fileskip(filecount, samples_per_file)

ftab.meta['fileskip'] = fileskip

##########################################################################
# Loop through frames of data and split into channels
# +
# Save intensity stream
################################################################

t0 = time.time()

Nread = samples_per_file * fileskip

i_start = 0
i_end = i_start + int(Nread / 100)
stepcount = 0

for band in bands:
    band.seek(0)

for _ in range(int(filecount/fileskip)):

    subframe = np.zeros((Nread, Nchan, Npol), dtype=bands[0].dtype)

    for i in range(Nband):

        subframe[:, bw*i:bw*(i+1), :] = bands[i].read(Nread).reshape(-1, bw, Npol)

    subframe = np.sum(subframe, axis=2)
    subframe, _, _, _ = imbin(subframe, binning_r=binning)

    i_stream[i_start:i_end] = subframe
    i_start = i_end
    i_end = i_start + int(Nread / 100)

    del subframe

    t = time.time() - t0
    hh = int(t / 3600)
    mm = int((t % 3600)/60)
    ss = int(t % 60)

    stepcount += fileskip
    print(f'Read to file {stepcount}/{filecount}: {stepcount*100/(filecount):.2f}% complete; Time elapsed = {hh:02d}:{mm:02d}:{ss:02d}', end='               \r')

##########################################################################
# Close data streams
##########################################################################

for i in range(Nband):
    bands[i].close()
    data[i].close()

print('vdif data closed')

##########################################################################
# Write info table
##########################################################################

ftab.write(istream + 'SplitTab.hdf5', path='vdif_info', overwrite=True)

print('info table written')
