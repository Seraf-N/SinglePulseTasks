import numpy as np
from astropy.table import QTable
import astropy.units as u

from numpy.lib.format import open_memmap
import os
import matplotlib.pyplot as plt

from utils import DispersionMeasure, imshift

import sys
if '/home/serafinnadeau/Python/packages/scintillometry/' not in sys.path:
    sys.path.append('/home/serafinnadeau/Python/packages/scintillometry/')

from scintillometry.shaping import ChangeSampleShape
from baseband import vdif

codedir = '/home/serafinnadeau/Scripts/Chime_Crab_GP/chime_crab_gp/'
plotdir = '/home/serafinnadeau/Plots/'
archdir = '/drives/STOPGAP/9/crab_archive/'
datadir = '/drives/STOPGAP/'

#date = 20190813
#if date:
#    archdir = archdir + f'{date}/'
#    istream = archdir + 'istream/'
#    splitdir = archdir + 'splitdir/'
#    pulsedir = archdir + 'pulsedir/'

def crab_nu_fit(tab, start=29.5, end=29.63, N=1000, plot=False):
    '''
    From list of GP detections, calculates the pulse period by wrapping
    the detection times by the 1/(pulse frequency) for a range of spin 
    frequencies, and then choosing the best guess value by finding which
    guess has the highest standard deviation in the change of the cumulative
    detections over the phase.

    
    '''
    ptab = tab.copy()#QTable(np.copy(ptab))
    ptab.sort('off_s')
    time = ptab['off_s']
    snr = ptab['snr']

    I = []
    nus = np.linspace(start, end, N)
    phase = np.linspace(0, 1, N)

    i = 1
    for nu in nus:
        # Wrap the detection time uising the guess frequency to get a phase
        pwrap = (time.value / (1 / nu)) % 1
        # Get the cumulative number of detections as a function of phase
        count = []
        for phi in phase:
            m = pwrap < phi
            count += [np.sum(m)*1.]
        # Get the change in counts as a function of phase
        delta = count - np.roll(count, 1)
        delta[0] = np.nan
        # Characterize how good an approximation nu is by the standard deviation
        #of delta across phase
        I += [np.nanstd(delta)]
        
        if plot:
            if i == 0:
                plt.figure('Phase')
                plt.xlabel('Phase [cycle]')
                plt.ylabel('Detection S/N')
                plt.figure('Cumulative Index')
                plt.xlabel(r'Phase $\phi$ [cycle]')
                plt.ylabel('Cumulative number of detections N')
                plt.figure('Change in Cumulative Index')
                plt.xlabel(r'Phase $\phi$ [cycle]')
                plt.ylabel(r'$\frac{\Delta N}{\Delta \phi}$')

            if (i*100/N)%10 == 0:
                plt.figure('Phase')
                plt.plot(pwrap, time, '.', alpha=0.3)
                plt.figure('Cumulative Index')
                plt.plot(phase, count, alpha=0.3, label=f'{nu:.4f}')
                plt.figure('Change in Cumulative index')
                plt.plot(phase, delta, alpha=0.3, label=f'{nu:.4f}')
            i += 1
    
    # Fit I for the pulsar frequency
       
    if plot:
        plt.figure('Standard deviation in change of counts')
        plt.xlabel(r'$\nu$ [Hz]')
        plt.ylabel(r'Stadard deviation in $\frac{\Delta N}{\Delta \phi}$')
        plt.plot(nus, I)
            
    ########################################################################
    # Return estimate of nu. Needs to be made more rigorious by fitting for 
    # it in future, with error above and below estimate. For now, good guess.
    ########################################################################
    error = (end - start) / (N - 1)
    return nus[np.argmax(I)], error

def crab_nu_fit_iter(ptab, start=29.6, end=29.63, N=1000, plot=False):
    '''
    More efficient version of crab_nu_fit which iterates over guesses made
    with lower N until N guesses made
    Note: plot does not work properly
    '''

    def iter(ptab, start, end, N):
        '''
        '''
        time = ptab['off_s']
        snr = ptab['snr']

        nus = np.linspace(start, end, N/10)
        dnu = (end - start) / (N/10 - 1)
        phase = np.linspace(0, 1, 1/dnu)

        I = []
        
        for nu in nus:
            pwrap = np.array(time) % (1/nu)
            count = []
            for phi in phase:
                m = pwrap < phi
                count += [np.sum(m)*1.]
            delta = count - np.roll(count, 1)
            delta[0] = np.nan
            I += [np.nanstd(delta)]        

        nu = nus[np.argmax(I)]

        return nu, dnu, nus, I, phase, count, snr, pwrap    

    nu, dnu, nus, I, phase, count, snr, pwrap = iter(ptab, start, end, N)

    Nus = list(nus)
    I = list(I)
    n = N/10
    Phase = [phase]
    Count = [count]
    Snr = [snr]
    Pwrap = [pwrap]

    while n <= N:
        print(n)
        start = nu - 3*dnu
        end = nu + 3*dnu
        nu, dnu, nus, i, phase, count, snr, pwrap = iter(ptab, start, end, N)

        Nus += list(nus)
        I += list(i)
        Phase += [phase]
        Count += [count]
        Snr += [snr]
        Pwrap += [pwrap]

        n += N/10

    if plot:
        plt.figure('Standard deviation in change of counts')
        plt.xlabel(r'$\nu$ [Hz]')
        plt.ylabel(r'Stadard deviation in $\frac{\Delta N}{\Delta \phi}$')
        plt.plot(Nus, I)

        plt.figure('Phase')
        plt.xlabel('Phase [cycle]')
        plt.ylabel('Detection S/N')

        plt.figure('Cumulative Index')
        plt.xlabel(r'Phase $\phi$ [cycle]')
        plt.ylabel('Cumulative number of detections N')

        for i in range(10):
            plt.figure('Phase')
            plt.plot(Pwrap[i], Snr[i])

            plt.figure('Cumulative Index')
            plt.plot(Phase[i], Count[i])

    return nu, dnu

def read_pulse(date, number, directory='/drives/CHF/9/crab_archive/'):
    '''
    Reads a specific archived pulse.
    '''
    pulsedir = f'{directory}{date}/pulsedir/'#f'/drives/CHF/9/crab_archive/{date}/pulsedir/'

    pulse = open_memmap(pulsedir + f'pulse_{number:04d}.npy', dtype=np.float16, 
                        mode='r', shape=(15625, 1024, 4))

    return pulse

def DM_alter(I, dm_new, dm_old=56.7):
    '''
    Modifies the DM of the pulse and outputs the altered waterfall 
    '''
    dm_delta = DispersionMeasure(dm_old - dm_new)

    freqs = np.linspace(800, 400, 1024) * u.MHz
    reffreq = 800 * u.MHz    

    time_delay = dm_delta.time_delay(freqs, reffreq)

    pixel_delay = time_delay.value / 2.56e-6

    if len(np.shape(I)) == 3:

        I_alt = np.zeros(np.shape(I))

        for i in range(np.shape(I)[2]):
            pol = imshift(I[:,:,i], shiftr=pixel_delay)
            I_alt[:,:,i] = pol

    else:
        I_alt = imshift(I, shiftr=pixel_delay)
    
    return I_alt

def read_stream(date):
    '''
    Reads the un-dedispersed, binned intensity stream for the input date
    '''
    streamdir = f'{archdir}{date}/istream/'

    I = open_memmap(streamdir + 'i_stream.npy', mode='r+')

    return I

def readpulse(date, number, dm=56.7, prepulse=2625, profilewidth=15625, subpix=True):
    '''
    Reads and dedisperses single pulse from raw data using homebrew code, 
    instead of baseband -- meant to test dedispersion issue
    '''

    dm = DispersionMeasure(dm)
    DP = []

    istream = f'{archdir}{date}/istream/'
    slpitdir = f'{archdir}{date}/splitdir/'

    tab = QTable.read(istream + 'pulse_tab.fits')
    m = tab['fname'] == f'pulse_{number:04d}.npy'
    pos = int(tab[m]['pos'])
    TZ = tab.meta['HISTORY'][1][19:-18]
    binning = tab.meta['BINNING']
    tbin = u.Quantity(tab.meta['TBIN'])
    pos = pos * binning

    datadir = '/drives/CHA/'
    x = get_vdif_files(date, TZ, datadir)

    pulse = np.zeros((profilewidth, 1024, 4))
    f = np.linspace(800, 400, 1024)
    
    index = 0

    for chan in range(1024):

        if index % 8 == 0:
            if index != 0:
                frame.close()
            fname = splitdir + f'Split_Channel_c{index:04d}-{index+7:04d}.vdif'
            frame = vdif.open(fname, 'rs', sample_rate=1/(2.56*u.us))
            chanframe = ChangeSampleShape(frame, reshape)
            
        dt = dm.time_delay(f[chan]*u.MHz, 800*u.MHz)
        dp = dt.value / tbin.value
        start = pos - 2625 + int(dp)
        print(start)
        end = start + profilewidth

        chanframe.seek(start*2.56e-6*u.s)
        pulsechan = chanframe.read(profilewidth)[:, int(index%8), :]

        pulse[:,chan,0] = np.real(pulsechan[:,0])
        pulse[:,chan,1] = np.imag(pulsechan[:,0])
        pulse[:,chan,2] = np.real(pulsechan[:,1])
        pulse[:,chan,3] = np.imag(pulsechan[:,1])
        
        DP += [dp]

    if subpix:
        subpix = np.array(DP) % 1
        rex = pulse[:,:,0]
        imx = pulse[:,:,1]
        rey = pulse[:,:,2]
        imy = pulse[:,:,3]
        rex = imshift(rex, shiftr=subpix, realout=False)
        imx = imshift(imx, shiftr=subpix, realout=False)
        rey = imshift(rey, shiftr=subpix, realout=False)
        imy = imshift(imy, shiftr=subpix, realout=False)
        pulse[:,:,0] = rex
        pulse[:,:,1] = imx
        pulse[:,:,2] = rey
        pulse[:,:,3] = imy

    return pulse

def reshape(data):
    reshaped = data.transpose(0, 2, 1)    
    return reshaped

def get_vdif_files(date, time, datadir=datadir):
    '''
    '''
    fnames = []
    N = os.listdir(datadir)
    for i in N:
        X = os.listdir(datadir+f'{i}/{date}T{time}Z_chime_psr_vdif/')
        X.sort()
        X = X[:-1]
        for x in X:
            fnames += [datadir+f'{i}/{date}T{time}Z_chime_psr_vdif/' + x]
        
    fnames.sort()
    return fnames


def dedisperse_chunk(chunk, dm, s_per_sample=2.56e-6):
    '''
    Dedisperses the chunk with the given dm
    '''
    freqs = np.linspace(800, 400, 1024, endpoint=False) * u.MHz
    dt = dm.time_delay(800*u.MHz, freqs)

    chunk_dd = imshift(chunk*1, shiftr=dt.value/s_per_sample)

    return chunk_dd
    

def dm_fit(I, DM0, s_per_sample=2.56e-6, Niter=3, L=50):
    
    DM0 = DispersionMeasure(DM0)
        
    dms = DispersionMeasure(np.linspace(56.6, 56.9, L))
    ddm = (dms[0] - dms[-1])/len(dms)
    DMs = []
    S = []
    
    binning = s_per_sample / 2.56e-6
    
    bound = 0.02/s_per_sample
    
    Iter = 1
    
    while Iter <= Niter:
        
        count = 0
    
        for dm in dms:
            I_shift = dedisperse_chunk(I, dm-DM0, s_per_sample)
            i = np.sum(I_shift, axis=1)
            m = np.median(i)
            M = np.max(i - m)

            max_bound = int(np.argmax(i - m)+bound)

            N = np.nanstd(i[max_bound:] - m)

            DMs += [dm]
            S += [M / N]

            print(f'Scan {Iter}: {(count+1)*100/len(dms):.2f}% Complete', end='            \r')

            count += 1


        count = 0
        pos = np.argmax(S)
        dm_peak = DMs[pos].value
        
        Iter += 1   
        dms = DispersionMeasure(np.linspace(dm_peak-0.1**Iter * Iter, dm_peak+0.1**Iter * Iter, L))
       
    
    return DMs, S

def crab_nu(tab, nu0=29.6, width=0.1, N=1000, it=6, bins=100, plot=False):
    
    time = tab['off_s'].value
    
    counts = []
    nus = []
    
    for i in range(it):
    
        nu_test = np.linspace(nu0-width/2**(i+1), nu0+width/2**(i+1), N+1)

        for nu in nu_test:
            
            if nu not in nus:
                
                nus += [nu]
                h = np.histogram(time % (1 / nu), bins=bins)
                counts += [np.max(h[0])]
                
        nu0 = nus[np.argmax(counts)]
     
    nu_max = nus[np.argmax(counts)]
    
    t = QTable()
    t['nu'] = nus
    t['count'] = counts
    t.sort('nu')
    
    if plot:
        plt.plot(t['nu'], t['count'])
        plt.xlim(nu_max-0.001, nu_max+0.001)
        
    return nu_max, t['nu'], t['count']
