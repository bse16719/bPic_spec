#Import library and packages, modules
import sys
import astropy.io.fits as fits
import astropy.table as table
import numpy as np
import scipy
import scipy.optimize
import scipy.signal
import scipy.ndimage
from scipy.special import wofz
import pandas as pd
import astropy.io.fits as fits
import pylab as plt
import numpy as np
import scipy as sp

from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from PySpectrograph.Spectra.Spectrum import air2vac
    
from astropy.time import Time
from astropy.coordinates import SkyCoord, solar_system, EarthLocation, ICRS, UnitSphericalRepresentation, CartesianRepresentation
from astropy import coordinates

from astropy.coordinates import SkyCoord, solar_system, EarthLocation, ICRS, UnitSphericalRepresentation, CartesianRepresentation
from astropy import coordinates

from astropy.io import fits
import numpy as np
import glob
from pylab import *

import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter

from astropy.io import fits as pyfits

from sklearn.preprocessing import normalize


#uploads files here
#loop over each image and read in the data
infile_list = glob.glob('mean*_spec.fits')
infile_list.sort()

#select order to be shown
sel_order = sys.argv[1]

# convert wavelength to RV space
#====================================================================================
def Wave2RV(Wave,rest_wavelength,RV_BP):
    c= 299792458 # m/s
    #RV_BP = 20.5 km/s
# Convert to beta pic reference frame
    obs_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength
    delta_wavelength = Wave - obs_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)*1.e-3  # km/s
    return RV
#====================================================================================
  

# low order poly fit to each order of spectrum
#=====================================================================================
def polyfitr(x, y, order, clip, xlim=None, ylim=None, mask=None, debug=False):
    """ Fit a polynomial to data, rejecting outliers.

    Fits a polynomial f(x) to data, x,y.  Finds standard deviation of
    y - f(x) and removes points that differ from f(x) by more than
    clip*stddev, then refits.  This repeats until no points are
    removed.

    Inputs
    ------
    x,y:
        Data points to be fitted.  They must have the same length.
    order: int (2)
        Order of polynomial to be fitted.
    clip: float (6)
        After each iteration data further than this many standard
        deviations away from the fit will be discarded.
    xlim: tuple of maximum and minimum x values, optional
        Data outside these x limits will not be used in the fit.
    ylim: tuple of maximum and minimum y values, optional
        As for xlim, but for y data.
    mask: sequence of pairs, optional
        A list of minimum and maximum x values (e.g. [(3, 4), (8, 9)])
        giving regions to be excluded from the fit.
    debug: boolean, default False
        If True, plots the fit at each iteration in matplotlib.

    Returns
    -------
    coeff, x, y:
        x, y are the data points contributing to the final fit. coeff
        gives the coefficients of the final polynomial fit (use
        np.polyval(coeff,x)).

    Examples
    --------
    >>> x = np.linspace(0,4)
    >>> np.random.seed(13)
    >>> y = x**2 + np.random.randn(50)
    >>> coeff, x1, y1 = polyfitr(x, y)
    >>> np.allclose(coeff, [1.05228393, -0.31855442, 0.4957111])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, order=1, xlim=(0.5,3.5), ylim=(1,10))
    >>> np.allclose(coeff, [3.23959627, -1.81635911])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, mask=[(1, 2), (3, 3.5)])
    >>> np.allclose(coeff, [1.08044631, -0.37032771, 0.42847982])
    True
    """

    x = np.asanyarray(x)
    y = np.asanyarray(y)
    isort = x.argsort()
    x, y = x[isort], y[isort]

    keep = np.ones(len(x), bool)
    if xlim is not None:
        keep &= (xlim[0] < x) & (x < xlim[1])
    if ylim is not None:
        keep &= (ylim[0] < y) & (y < ylim[1])
    if mask is not None:
        badpts = np.zeros(len(x), bool)
        for x0,x1 in mask:
            badpts |=  (x0 < x) & (x < x1)
        keep &= ~badpts

    x,y = x[keep], y[keep]
    if debug:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y,'.')
        ax.set_autoscale_on(0)
        pl.show()

    coeff = np.polyfit(x, y, order)
    if debug:
        pts, = ax.plot(x, y, '.')
        poly, = ax.plot(x, np.polyval(coeff, x), lw=2)
        pl.show()
        raw_input('Enter to continue')
    norm = np.abs(y - np.polyval(coeff, x))
    stdev = np.std(norm)
    condition =  norm < clip * stdev
    y = y[condition]
    x = x[condition]
    while norm.max() > clip * stdev:
        if len(y) < order + 1:
            raise Exception('Too few points left to fit!')
        coeff = np.polyfit(x, y, order)
        if debug:
            pts.set_data(x, y)
            poly.set_data(x, np.polyval(coeff, x))
            pl.show()
            raw_input('Enter to continue')
        norm = np.abs(y - np.polyval(coeff, x))
        stdev = norm.std()
        condition =  norm < clip * stdev
        y = y[condition]
        x = x[condition]

    return coeff,x,y
#=====================================================================================
# fitting, outliers rejection and set poly order 
sigma_clip = 3.3
polyfit_order = 6
num_polyfits = 2

# find a reference spectrum
hdu_ref = pyfits.open('../sd_20161103/raw/vpH201611030023_spec.fits') 
wave_ref = hdu_ref[1].data['Wavelength']
flux_ref = hdu_ref[1].data['Flux']
order_ref = hdu_ref[1].data['Order']
pupsta_ref = hdu_ref[0].header['PUPSTA'] 

for i, img in enumerate(infile_list):
    
    
    hdu = fits.open(img)
    wave  = hdu[1].data['Wavelength']
    flux  = hdu[1].data['Flux']
    order = hdu[1].data['Order']
    thisspec_name = hdu[0].header['OBJECT']
    date_obs = hdu[0].header['DATE-OBS']
    
    # RV space 
    R_v = Wave2RV(wave, 3968.47,20.00) #3933.66, 3968.4673, 
    #print('RV :', R_v)

    try:
    	pupsta = hdu[0].header['PUPSTA']
    except KeyError:
        print("{} has no PUPSTA".format(img))
        pupsta = 1.0
    #print(img, pupsta)
    # pupend is missing from the headers and to be added

    for o in sys.argv[1:]:
       o = int(o)
       mask = (order==o)

       flux[mask]= np.convolve(flux[mask]/pupsta, np.ones(1), 'same')/np.median(np.convolve(flux[mask]/pupsta, np.ones(1), 'same'))

       #reference spectra
       flux_ref[mask] = np.convolve(flux_ref[mask]/pupsta_ref, np.ones(1), 'same')/np.median(np.convolve(flux_ref[mask]/pupsta_ref, np.ones(1), 'same'))     

       # mean flux
       avg_flux=sum(flux[mask])/float(len(flux[mask]))
       #print('mean flux:', avg_flux)

       # median flux
       med_flux=np.median(flux[mask])
       #print('median flux:', med_flux)


       # std flux
       std_flux=np.std(flux[mask]-avg_flux)
       #print('stdev flux:', std_flux)

# blaze correction for the continuum matching of spectra

# relative flux
       rel_flux = flux / flux_ref
       rel_flux[mask] = flux[mask] / flux_ref[mask]
    
#copy data for blaze correction  
       pyhrs_wavelength = wave[mask]
       pyhrs_fluxlvl = rel_flux[mask]
       pyhrs_fluxlvl_org = flux[mask]

# fitting poly to reference spectrum
       py_coeff, py_C_Wave, py_C_offsets = polyfitr(pyhrs_wavelength, pyhrs_fluxlvl, order=polyfit_order, clip=sigma_clip)
       py_p = np.poly1d(py_coeff)
       xs = np.arange(min(pyhrs_wavelength), max(pyhrs_wavelength), 0.1)
       ys = np.polyval(py_p, xs)

# blaze correction 
       final_pyhrs_flux = pyhrs_fluxlvl_org/np.polyval(py_p, pyhrs_wavelength)

       #plot data 
       #plt.plot(wave[mask],  flux[mask], ls='-', linewidth=2.5, color=c, label="SALT-HRS Obs block: ({0})".format(date_obs))  #

       #plt.plot(R_v[mask],  flux[mask], ls='-', linewidth=2.5, color=c, label="SALT-HRS Obs block: ({0})".format(date_obs))

#set colors for many plots
       ax1 = plt.subplot
       num_spectra = 22
       colours = plt.cm.coolwarm(np.linspace(0.0,1.0,num_spectra))

#plot data in Wavelength space
#       plt.plot(wave[mask],  flux[mask],color=colours[i],label="SALT-HRS Obs block: ({0})".format(date_obs)) 
#plot data in RV space
#       plt.plot(R_v[mask],  flux[mask],color=colours[i],label="SALT-HRS Obs block: ({0})".format(date_obs)) 

# blaze corrected flux
       plt.plot(pyhrs_wavelength,   final_pyhrs_flux,color=colours[i],label="SALT-HRS Obs block: ({0})".format(date_obs)) 
        
       plt.title(thisspec_name)
       plt.xlabel('Wavelength (Angstroms) [air.heliocentric]')

       # RV Space       
       #plt.xlabel('Radial Velocity (km/s) [air.heliocentric]')
       plt.ylabel('Normalised Flux (Arb. units) [Median]')
      
'''      
       wv_start = 3931.00  # CaII H-line=3967.42
       wv_end = 3936.00    # CaII H-line=3970.73  
 
       plt.xlim(wv_start,wv_end) 
       #plt.ylim(0.0,2.5)
       window = np.where((pyhrs_wavelength >= wv_start) & (pyhrs_wavelength< wv_end))
       thiswvs =pyhrs_wavelength[window]
       thisflux = final_pyhrs_flux[window]
       xlen = np.arange(min(thiswvs), max(thiswvs), 0.1) 
# ROI 
       ax2 = plt.subplot(111)
       
       ax2.plot(thiswvs, thisflux,color=colours[i],label="SALT-HRS Obs block: ({0})".format(date_obs)) 
       #plt.xticks(xlen,thiswvs) 
       ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
'''


# line features identified guide 
# H-delta line: 4861.34
#plt.axvline(4101.75, color='k', linestyle='--', linewidth=2.0, label='H delta-Line position= 4101.75 A')
# H-beta line: 4861.34
#plt.axvline(4861.34, color='k', linestyle='--', linewidth=2.0, label='H beta-Line position= 4861.34 A')
#CaII H and K lines
#plt.axvline(3968.66, color='k', linestyle='--', linewidth=2.0, label='CaII H-Line position= 3968.66 A')
#plt.axvline(3933.55, color='k', linestyle='-.', linewidth=2.0, label='CaII K-Line position= 3933.55 A')

# lines by Ersnt to monitor
'''plt.axvline(4041.35, color='k', linestyle='--', label='Line position= 4041.35 A') #order 116
plt.axvline(4071.74, color='k', linestyle='--', label='Line position= 4071.74 A') # 115
plt.axvline(4132.10, color='k', linestyle='--', label='Line position= 4132.10 A')
plt.axvline(4167.27, color='k', linestyle='--', label='Line position= 4167.27 A')
plt.axvline(4246.84, color='k', linestyle='--', label='Line position= 4246.84 A')
plt.axvline(4394.86, color='k', linestyle='--', label='Line position= 4394.86 A')
plt.axvline(4481.27, color='k', linestyle='--', label='Line position= 4481.27 A')
plt.axvline(4501.27, color='k', linestyle='--', label='Line position= 4501.27 A')
plt.axvline(4515.33, color='k', linestyle='--', label='Line position= 4515.33 A')
plt.axvline(4571.97, color='k', linestyle='--', label='Line position= 4571.97 A')
plt.axvline(4702.99, color='k', linestyle='--', label='Line position= 4702.99 A')
plt.axvline(4714.42, color='k', linestyle='--', label='Line position= 4714.42 A')
plt.axvline(4805.09, color='k', linestyle='--', label='Line position= 4805.09 A')
plt.axvline(4824.24, color='k', linestyle='--', label='Line position= 4824.24 A') #order 96
plt.axvline(4957.60, color='k', linestyle='--', label='Line position= 4957.60 A') #order 94
plt.xlim(3925,5050) #lines of interest
plt.ylim(0.0,1.5)
'''

#plt.legend(loc='best')
plt.show()
       


