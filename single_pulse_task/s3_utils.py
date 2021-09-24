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

def open_data(banddir, tab, Nband, startskip=0, endskip=1):
    
    data = []
    
    for i in range(Nband):
        mask = tab['fband'].astype(int) == i
        files = tab[mask]['fname']
        
        paths = []

        for f in files:
            paths += [banddir + f]
        
        data += [vdif.open(paths[startskip:-endskip], 'rs', sample_rate=1/(tab.meta['s_per_sample']*u.s), verify=False)]
        
    return data

def reshape(data_raw):
    return data_raw.reshape((-1, 128, 2))

def dedisperse_subbands(data, tab, dm):
    
    Nband = tab.meta['subbands']
    Nchan = tab.meta['nchan']
    Npol = tab.meta['npol']
    bw = int(Nchan / Nband)
    
    dedispersed = []
    freqs = np.tile(np.linspace(800, 400, Nchan, endpoint=False)[:,np.newaxis], (1, Npol))
    
    if type(dm) != u.quantity.Quantity:
        
        dm = dm * u.pc / u.cm**3
    
    for i in range(Nband):
        
        f = freqs[bw*i:bw*(i+1)]
        
        reshaped = ChangeSampleShape(data[i], reshape)
        dedispersed += [Dedisperse(reshaped, dm, frequency=f*u.MHz, sideband=1, reference_frequency=800*u.MHz)]
        
    return dedispersed

def read_pulse(dedispersed_data, tab, pos, prepulse, Nread):
    
    start_skip = tab.meta['start_skipped']
    start_time = tab.meta['start_time']
    samples_per_file = tab.meta['samples_per_file']
    
    tbin = tab.meta['s_per_sample'] * u.s
    binning = tab.meta['binning']
    Nband = tab.meta['subbands']
    Nchan = tab.meta['nchan']
    Npol = tab.meta['npol']
    bw = int(Nchan / Nband)
    
    pulse = np.zeros((Nread, 1024, 4), dtype=np.float16)
    
    pulse_time = (pos * binning - prepulse + start_skip*samples_per_file) * tbin
    freqs = np.linspace(800, 400, Nchan, endpoint=False)
    
    start_time0 = dedispersed_data[0].start_time
    
    for i in range(Nband):
        
        frame = dedispersed_data[i]
        
        dstart = (frame.start_time - start_time0).to(u.s)
        dtau = frame.dm.time_delay(freqs[bw*i:bw*(i+1)]*u.MHz, freqs[0]*u.MHz)
        corr = dtau[0] - dstart
        
        frame.seek(pulse_time - dtau[0] + corr)
        readpulse = frame.read(Nread).reshape(-1, bw, Npol) 
        
        pulse[:,bw*i:bw*(i+1),0] = np.real(readpulse[:,:,0])
        pulse[:,bw*i:bw*(i+1),1] = np.imag(readpulse[:,:,0])
        pulse[:,bw*i:bw*(i+1),2] = np.real(readpulse[:,:,1])
        pulse[:,bw*i:bw*(i+1),3] = np.imag(readpulse[:,:,1])    
        
        del readpulse
        
    return pulse
