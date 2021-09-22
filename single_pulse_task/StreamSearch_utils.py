import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.table import QTable

import sys
if '/home/serafinnadeau/Python/packages/scintillometry/' not in sys.path:
    sys.path.append('/home/serafinnadeau/Python/packages/scintillometry/')

from numpy.lib.format import open_memmap
import os

import time

from utils import DispersionMeasure, imshift

def master(stream2D, w=31250, dm=56.7, s_per_sample=2.56e-4, verbose=True, prometheus=False):
    '''
    Takes a memmap 2D stream and converts it to a 1D dedispersed intensity 
    stream.
    It does this chunk by chunk, with the time width of each chunk defined by
    the input parameter w and dedisperses to the input dm parameter.
    '''

    # Obtain data for w samples at a time, accounting for lost samples due to 
    # dedispersion.
    dm = DispersionMeasure(dm)
    dt = dm.time_delay(800*u.MHz, 400*u.MHz)

    time_waste = int(abs(dt.value / s_per_sample) + 1)
    print(f'{time_waste} samples lost at the end of array due to dedispersion')

    w_eff = w - time_waste # the effective width of each read chunk after dedispersion

    N = int(len(stream2D) / w_eff) # the number chunks to be read in

    stream1D = np.zeros(N * w_eff)
    mask1D = np.zeros(N * w_eff)

    if verbose:
        t0 = time.time()
        chunk_n = -1
        verbose_print(chunk_n, N, t0, extra='', prometheus=prometheus)

    for chunk_n in range(N):
        
        sample_min = select_chunk(chunk_n, w_eff) # Calculate starting time bin of chunk

        if verbose:
            verbose_print(chunk_n, N, t0, extra='Reading')
        chunk = read_chunk(stream2D, w, sample_min) # Read in chunk
        
        if chunk_n == 0:
            rfi_mask = rfi_bands(chunk.T - np.mean(chunk.T, axis=0, keepdims=True))
            print('')
            print(np.sum(rfi_mask))
            print('')
            

        #if verbose:
        #    verbose_print(chunk_n, N, t0, extra='Masking')
        #chunk = mask_chunk(chunk)

        if verbose:
            verbose_print(chunk_n, N, t0, extra='Dedispersing')
        chunk = dedisperse_chunk(chunk, dm, s_per_sample)

        if verbose:
            verbose_print(chunk_n, N, t0, extra='Adding', prometheus=prometheus)
        stream1D = fuse_chunk(stream1D, chunk, sample_min, w_eff, rfi_mask)

    if verbose:
        verbose_print(chunk_n, N, t0, extra='Complete', rewrite=False, prometheus=prometheus)
        print('')

    return stream1D

def verbose_print(current, total, t0, extra='', prometheus=False, rewrite=True):
    tf = time.time()
    ss = int(tf-t0)
    hh = ss // 3600
    mm = ss // 60 - hh*60
    ss = ss % 60
    if rewrite:
        end = '                          \r'
    else:
        end = '                          \n'
    print(f'{current+1:0{len(str(total))}}/{total}: '
          f'{(current+1)*100/total:05.2f}% complete'
          f' -- {hh:02d}:{mm:02d}:{ss:02d} elapsed -- {extra: <20} ',
          end=end)
    
    if prometheus:
        import prometheus_client as pc
        
        stage, promdir, registry, completion, timing = prometheus
        completion.labels(stage=stage).set((current+1)*100/total)  
        timing.labels(stage=stage).set((tf - t0)/3600)  
        pc.write_to_textfile(promdir + f'crab_monitoring_{stage}.prom', registry)   
        
    return

def rollingmean(x, w, edge=False):
    roll = np.convolve(x, np.ones((w,))/w, mode='same')
    
    if edge:
        roll[:edge] = roll[edge+1]
        roll[-edge:] = roll[-edge-1]
    
    return roll

def select_chunk(chunk_n, w_eff):
    '''
    Calculates the starting sample of the memmaped 2D stream for a given chunk
    number.
    '''
    sample_min = w_eff * chunk_n
    return sample_min

def read_chunk(stream2D, w, sample_min):
    '''
    Reads in a time chunk of width w time bins, starting at sample_min from a 
    2D memmaped stream.
    Reshapes to (FREQ, TIME) / (1024, w)
    '''
    #stream2D.seek(sample_min)
    #chunk = stream2D.read(w)
    chunk = stream2D[sample_min:sample_min+w] * 1
    shape = np.shape(chunk)
    if shape[1] == 1024:
        chunk = chunk.transpose(1,0)

    return chunk

def mask_chunk(chunk):
    '''
    Replaces zero masking of RFI with the mean of the relevant frequency channel
    '''
    for i in range(len(chunk)): # Loop over frequencies
        row2 = chunk[i] * 1
        m = (chunk[i] == 0) + np.isnan(chunk[i])
        if np.sum(m) != 0:
            row2[m] = np.nan            # Mask all true zeros to nan
            mean = np.nanmean(row2)      # Compute the mean of each channel
            if np.isnan(mean):
                chunk[i][m] = 0 # if channel mean is nan, the fill channel back with 0
            else:
                chunk[i][m] = mean # Fill gaps in channel with the channel mean value
        else:
            chunk[i] = chunk[i]
    return chunk

def dedisperse_chunk(chunk, dm, s_per_sample):
    '''
    Dedisperses the chunk with the given dm
    '''
    
    freqs = np.linspace(800, 400, 1024, endpoint=False) * u.MHz
    dt = dm.time_delay(800*u.MHz, freqs)

    chunk = imshift(chunk*1, shiftc=dt.value/s_per_sample)

    return chunk

def fuse_chunk(stream1D, chunk, sample_min, w_eff, rfi=False):
    '''
    Collapses chunk and adds it to the dedispersed 1D stream
    '''
    if type(rfi) == np.ndarray:
        chunk[rfi,:] = 0
    chunk = chunk - np.nanmean(chunk.astype(np.float32), axis=1, keepdims=True)
    
    stream = np.sum(chunk, axis=0)
    stream1D[sample_min:sample_min+w_eff] = stream[:w_eff]

    return stream1D

def correct_stream(stream1D, savedir, N=1000):
    '''
    flattens the background levels of the intensity stream to better pick out 
    giant pulses in the search by itteratively subtracting the rolling mean.
    Saves the corrected 1D stream as a memmap object.
    '''
    mean_test = np.nanmean(stream1D)
    std_test = np.nanstd(stream1D)
    snr_test = (stream1D-mean_test)/std_test

    rollmean = rollingmean(snr_test, N, edge=N)
    
    snr_test = snr_test - rollmean
    mean_test = np.nanmean(snr_test)
    std_test = np.nanstd(snr_test)
    snr_test = (snr_test-mean_test)/std_test

    rollmean = rollingmean(snr_test, N, edge=N)

    snr_test = snr_test - rollmean
    mean_test = np.nanmean(snr_test)
    std_test = np.nanstd(snr_test)
    snr_test = (snr_test-mean_test)/std_test

    rollmean = rollingmean(snr_test, N, edge=N)

    snr_test = snr_test - rollmean
    mean_test = np.nanmean(snr_test)
    std_test = np.nanstd(snr_test)
    snr_test = (snr_test-mean_test)/std_test

    stream1D = open_memmap(savedir+'istream_corr.npy', dtype=np.float32, mode='w+', 
                           shape=np.shape(snr_test))
    stream1D[:] = snr_test
    
    return stream1D

def streamsearch(stream1D, splittab, cutoff, banddir, savedir, datestr, timestr, 
                 Nmax=False, Nmin=1024, dm=DispersionMeasure(56.7), output=False):
    '''
    Searches the corrected 1D stream for signals stonger than 'cutoff' sigma
    '''
    POS = []
    SNR = []
    snr_search = stream1D * 1

    start_time = Time(splittab.meta['start_time'], format='isot', precision=9)
    n_files = splittab.meta['vdif_files']
    startskip = splittab.meta['start_skipped']
    endskip = splittab.meta['end_skipped']
    filecount = n_files - (startskip + endskip)
    n_frames = splittab.meta['frames_per_file'] * filecount
    samples_per_frame = splittab.meta['samples_per_frame']
    binning = splittab.meta['binning']
    s_per_sample = splittab.meta['s_per_sample'] * binning
    
    nsamples = n_frames * samples_per_frame

    pos = np.nanargmax(snr_search)
    signal = snr_search[pos]
    snr_search[pos-30:pos+30]= np.nan
    snr = (signal - np.nanmean(snr_search[pos-150:pos+150])) / np.nanstd(snr_search[pos-150:pos+150])

    i = 0
    t0 = time.time()

    snr_search[:int(1.11/2.56e-6/100)] = 0

    while (snr > cutoff) or (len(POS) < Nmin):
        if (len(POS) < Nmax) or (not Nmax):
            
            POS += [pos]
            SNR += [snr]

            snr_search[pos-30:pos+30] = 0

            pos = np.nanargmax(snr_search)
            signal = snr_search[pos]
            snr_search[pos-30:pos+30] = np.nan
            snr = (signal - np.nanmean(snr_search[pos-150:pos+150])) / np.nanstd(snr_search[pos-150:pos+150])

            i += 1
            t = time.time() - t0
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            #print(f'Intensity stream searched for pulses: {len(POS)} pulses found -- '
            #      f'S/N: {snr:.3f} -- POS: {pos*100*2.56e-6:.3f} -- Time elapsed: '
            #      f'{int(h):02d}:{int(m):02d}:{int(s):02d}', end='                     \r')

    print(f'Intenisty stream searched for pulses: {len(POS)} pulses found                             ')

    POS = np.array(POS)
    TIME_S = POS * s_per_sample
    SNR = np.array(SNR)
    MJD = start_time + TIME_S * u.s
    
    # Create Table of GPs to be saved

    tab = QTable()
    tab.meta = splittab.meta

    tab['time'] = (TIME_S * u.s + start_time).isot

    tab['off_s'] = TIME_S * u.s
    tab['pos'] = POS
    tab['snr'] = SNR
    tab.sort('pos')


    tab.meta['DM'] = dm.value
    tab.meta['binning'] = 100
    tab.meta['sigma'] = cutoff
    tab.meta['start'] = start_time.isot
    tab.meta['nsamples'] = nsamples
    tab.meta['history'] = ['Intensity stream i_stream.npy saved from ChannelSplit'
                           f' on vdif files {banddir}*/{datestr}T{timestr}'
                           'Z_chime_psr_vdif/*',
                           'i_stream.npy dedispersed and searched for giant pulses']

    tab.write(savedir+f'search_tab.hdf5', path='search_info', overwrite=True)

    if output:
        return tab
    return

def rfi_bands(data, Nsigma=4, plot=False):
    
    normdata = data.astype(np.float32) - np.mean(data.astype(np.float32), axis=0, keepdims=True)
    std = np.std(normdata.astype(np.float32), axis=0)
    x = np.arange(0, std.shape[0])
    
    normmedian = np.median(normdata.astype(np.float32), axis=0)
    
    rfi_mask = std == 0
    rfi_count = np.sum(rfi_mask)
    rfi_count_old = 0
    
    norm_temp = normmedian*1
    
    while rfi_count != rfi_count_old:
        
        norm_temp -= np.nanmedian(norm_temp)
        cut = np.nanstd(norm_temp)
        
        rfi_mask += (norm_temp > Nsigma * cut) + (norm_temp < -Nsigma * cut)
        norm_temp[rfi_mask] = np.nan
        
        rfi_count_old = rfi_count * 1
        rfi_count = np.sum(rfi_mask)
        
        if plot:
            
            plt.figure(figsize=[18,6])
            plt.plot(x[~rfi_mask], norm_temp[~rfi_mask])
            plt.hlines([-cut*Nsigma, cut*Nsigma], 0, 1024)
            plt.pause(0.1)
            
    return rfi_mask
