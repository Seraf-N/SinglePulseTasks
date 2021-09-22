import numpy as np
import time

import astropy.units as u

from scipy.stats import binned_statistic as bs

######################################################################################################################
######################################################################################################################

def imshift(im, shiftr=0, shiftc=0, realout=True, verbose=False):
    '''
        Shifts a 2D image by the given shift variables shiftr (rows, axis=0) and shiftc(columns, axis=1).
        
        shiftr / shiftc can be either int, float (scalar) or 1D numpy arrays
        
        If shiftr / shiftc are scalar, then all rows / columns will be shifted by the same value.
        
        If shiftr / shiftc are 1D arrays, then they must have the same length as the image rows / columns 
    and each row / column will be shifted by a different amount as specified by the inputed array.
    
        Outputs the shifted image. If realout = True, then only the real part of the ifft is outputed. 
    If the inputed image is real, then this the complex component of the ifft should be ~ 0.
    If instead realout = False, then the output will be given in is complex form.
    '''
    # Load row and column lengths of image
    Nr, Nc = np.shape(im)[:2]
    
    # Create row and column arrays for image
    r = np.linspace(0, Nr-1, Nr)
    c = np.linspace(0, Nc-1, Nc)
    
    # Convert from real, row / columns to fourier space row / columns 
    R = np.mod((r + Nr/2.),Nr) - Nr/2.
    C = np.mod((c + Nc/2.),Nc) - Nc/2.
    if len(np.shape(im)) == 3:
        rshape = (len(R), np.shape(im)[2])
        cshape = (len(C), np.shape(im)[2])
        R = np.zeros(rshape) + R[:,np.newaxis]
        C = np.zeros(cshape) + C[:,np.newaxis]
    
    # Make copy image to be shifted
    imshift = im*1

    #shift columns along row direction
    if verbose:
        t0 = time.time()
    
    if np.array(shiftr).any() != 0:
        for i in range(Nc):
        
            if type(shiftr) == np.ndarray:
                shift = shiftr[i]
            else:
                shift = shiftr
            
            im_i = imshift[:, i] * 1
        
            # Apply shift to row in Fourier space
            IM_i = np.fft.fft(im_i)
            IM_i *= np.exp(-1j * 2*np.pi * (shift*R/Nr))
        
            # Convert shifted row from Fourier space to real space
            im_i = np.fft.ifft(IM_i)
            imshift[:, i] = np.real(im_i)
        
            if verbose:
                tf = time.time()
                t = tf - t0
                hh = int(t/3600)
                mm = int((t % 3600)/60)
                ss = int(t % 60)
                print('Col Shift {}/{}: {}% complete'.format(i+1, Nc, 100*(i+1)/Nc), 'Time elapsed: {}h{}m{}s'.format(hh, mm, ss), end='                                         \r')

    if verbose:
        if np.array(shiftr).any() != 0:
            print('All columns shifted: {}h{}m{}s ellapsed'.format(hh, mm, ss), end='                                                         \n')
    
    if verbose:
        t0 = time.time()
    #shift rows along column direction
    if np.array(shiftc).any() != 0:
        for i in range(Nr):
        
            if type(shiftc) == np.ndarray:
                shift = shiftc[i]
            else: 
                shift = shiftc
            
            im_i = imshift[i]*1
        
            # Apply shift to column in Fourier space
            IM_i = np.fft.fft(im_i)
            IM_i *= np.exp(-1j * 2*np.pi * (shift*C/Nc))
        
            # Convert shifted column from Fourier space to real space
            im_i = np.fft.ifft(IM_i)
            imshift[i] = np.real(im_i)
        
            if verbose:
                tf = time.time()
                t = tf - t0
                hh = int(t/3600)
                mm = int((t % 3600)/60)
                ss = int(t % 60)
                print('Col Shift {}/{}: {}% complete'.format(i+1, Nr, 100*(i+1)/Nr), ' -- Time elapsed: {}h{}m{}s'.format(hh, mm, ss), end='                                         \r')

    if verbose:
        if np.array(shiftc).any() != 0:
            print('All rows shifted: {}h{}m{}s ellapsed'.format(hh, mm, ss), end='                                                         \n')
   
    # Output shifted image
    if realout:
        return np.real(imshift)
    
    return imshift

#########################################################################################################################
#########################################################################################################################

def imbin(im, binning_c=1, binning_r=1):
    """
    Bins image by binning_c in the column direction and binning_r in the row direction
    """
    
    shape = np.shape(im)
        
    from functools import reduce

    def factors(n):    
        return np.array(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    
    fac_r = factors(shape[0])
    fac_c = factors(shape[1])
    
    bin_r = fac_r[np.abs(fac_r-binning_r).argmin()]
    #print('row binning set to {}, closest factor of row length to {}'.format(bin_r, binning_r))
    bin_c = fac_c[np.abs(fac_c-binning_c).argmin()]
    #print('col binning set to {}, closest factor of col length to {}'.format(bin_c, binning_c))
    
    rows = np.arange(shape[0])
    if binning_r == 1:
        ROWS = rows
    else:
        ROWS = bs(rows, rows, statistic='mean', bins=len(rows)/bin_r)
        
    cols = np.arange(shape[1])
    if binning_c == 1:
        COLS = cols
    else:
        COLS = bs(cols, cols, statistic='mean', bins=len(cols)/bin_c)
        
    def rebin(arr, new_shape):
        if len(new_shape) == 2:
            shape = (int(new_shape[0]), int(arr.shape[0] // new_shape[0]),
                     int(new_shape[1]), int(arr.shape[1] // new_shape[1]))
        elif len(new_shape) == 3:
            shape = (int(new_shape[0]), int(arr.shape[0] // new_shape[0]),
                     int(new_shape[1]), int(arr.shape[1] // new_shape[1]), 
                     int(new_shape[2]), int(arr.shape[2] // new_shape[2]))
            
        return arr.reshape(shape).mean(-1).mean(1)
        
    if len(shape) == 2:
        binning = np.array([bin_r, bin_c])
    elif len(shape) == 3:
        binning = np.array([bin_r, bin_c, 1])
        
    IM = rebin(im, (shape / binning))
    
    return IM, ROWS[0], COLS[0], binning

###########################################################################################################
###########################################################################################################

class DispersionMeasure(u.quantity.SpecificTypeQuantity):
    # Constant hardcoded to match assumption made by tempo.
    # Taylor, Manchester, & Lyne 1993ApJS...88..529T
    dispersion_delay_constant = u.s / 2.41e-4 * u.MHz**2 * u.cm**3 / u.pc
    _equivalent_unit = _default_unit = u.pc / u.cm**3

    def time_delay(self, f, fref=None):
        """Time delay due to dispersion.

        Parameters
        ----------
        f : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        fref : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.
        """
        d = self.dispersion_delay_constant * self
        fref_inv2 = 0. if fref is None else 1. / fref**2
        return d * (1./f**2 - fref_inv2)

    def phase_delay(self, f, fref=None):
        """Phase delay due to dispersion.
  
        Parameters
        ----------
        f : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        fref : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.
        """
        d = self.dispersion_delay_constant * u.cycle * self
        fref_inv = 0. if fref is None else 1. / fref
        return d * (f * (fref_inv - 1./f)**2)

    def phase_factor(self, f, fref=None):
        """Complex exponential factor due to dispersion.

        This is just ``exp(1j * phase_delay)``.

        Parameters
        ----------
        f : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        fref : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.
        """
        return np.exp(self.phase_delay(f, fref).to(u.rad).value * 1j)
