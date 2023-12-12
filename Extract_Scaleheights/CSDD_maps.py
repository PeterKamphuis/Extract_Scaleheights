#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-


# This program makes CSDD maps as described in Olling 1995

import pk_common_functions.functions as cf
import support_functions as sf
import sys
import os
import copy
import numpy as np
import scipy.interpolate as spirp
import scipy.ndimage as ndimage
from optparse import OptionParser
from astropy.io import fits
from astropy.wcs import WCS
import warnings

def createCSDDmaps(argv):
    start_dir = os.getcwd()

    #Constants that are used in the code
    global H_0
    H_0 = 70. #km/s/Mpc #Hubble constant

    #Then check the input options
    parser  = OptionParser()
    parser.add_option('-c','--cube', action ="store" ,dest = "cube", default = '', help = 'Define the cube to be analysed.',metavar='CUBE')
    parser.add_option('-d','--deffile', action ="store" ,dest = "deffile", default = '', help = 'Input def file',metavar = 'DEFFILE')
    parser.add_option('-o','--csdd_cube', action ="store" ,dest = "csdd_cube", default='csdd_cube.fits', help = 'Name of the output CSDD cube',metavar='OUT_CUBE')
    input_parameters,args = parser.parse_args()


    variables_we_need = ['RADI','INCL','INCL_2','PA','PA_2','VROT','VROT_2','SBR','SBR_2','NDISKS','SDIS','SDIS_2','VSYS','XPOS','YPOS']
    #First we the tirific PARAMETERS
    profiles =  cf.load_tirific(input_parameters.deffile,Variables = variables_we_need)

    #Make sure that everything is filles also for single disks
    if float(profiles[variables_we_need.index('NDISKS')][0]) == 1:
        profiles[variables_we_need.index('INCL_2')] = profiles[variables_we_need.index('INCL')]
        profiles[variables_we_need.index('PA_2')] = profiles[variables_we_need.index('PA')]
        profiles[variables_we_need.index('VROT_2')] = profiles[variables_we_need.index('VROT')]
        profiles[variables_we_need.index('SBR_2')] = profiles[variables_we_need.index('SBR')]
        profiles[variables_we_need.index('SDIS_2')] = profiles[variables_we_need.index('SDIS')]

    #Create symmetric functions
    incl_r = spirp.interp1d(profiles[variables_we_need.index('RADI')],\
            [np.mean([x,y]) for x,y in zip(profiles[variables_we_need.index('INCL')],profiles[variables_we_need.index('INCL_2')])])
    pa_r = spirp.interp1d(profiles[variables_we_need.index('RADI')],\
            [np.mean([x,y]) for x,y in zip(profiles[variables_we_need.index('PA')],profiles[variables_we_need.index('PA_2')])])
    vrot_r = spirp.interp1d(profiles[variables_we_need.index('RADI')],\
            [np.mean([x,y]) for x,y in zip(profiles[variables_we_need.index('VROT')],profiles[variables_we_need.index('VROT_2')])])
    sbr_r = spirp.interp1d(profiles[variables_we_need.index('RADI')],\
            [np.mean([x,y]) for x,y in zip(profiles[variables_we_need.index('SBR')],profiles[variables_we_need.index('SBR_2')])])
    sdis_r = spirp.interp1d(profiles[variables_we_need.index('RADI')],\
            [np.mean([x,y]) for x,y in zip(profiles[variables_we_need.index('SDIS')],profiles[variables_we_need.index('SDIS_2')])])

    #open our dumb cube as a template
    hdr = fits.getheader(input_parameters.cube)
    if hdr['CDELT3'] < 100.:
        hdr['CDELT3'] = hdr['CDELT3']*1000.
        hdr['CRVAL3'] = hdr['CRVAL3']*1000.
        hdr['CUNIT3'] = 'M/S'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube_wcs = WCS(hdr)
    # make an empty fake cube centered on the galaxy
    out_header = copy.deepcopy(hdr)
    x_center,y_center,z_center = cube_wcs.wcs_world2pix(profiles[variables_we_need.index('XPOS')][0],\
                                           profiles[variables_we_need.index('YPOS')][0],\
                                           profiles[variables_we_need.index('VSYS')][0]*1000.,0.)

    out_header['CRVAL1'] = 0.
    out_header['CRPIX1'] = 1
    out_header['CTYPE1'] = 'Offset'
    out_header['CUNIT1'] = 'arcsec'
    out_header['CDELT1'] = float(abs(out_header['CDELT1'])*3600.)
    out_header['CRVAL2'] = 0.
    out_header['CRPIX2'] = 1
    out_header['CTYPE2'] = 'LOS Offset'
    out_header['CDELT2'] = float(abs(out_header['CDELT2'])*3600.)
    out_header['CUNIT2'] = 'arcsec'
    out_header['CRVAL3'] = 0.
    out_header['CRPIX3'] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csdd_wcs = WCS(out_header)
    #make an empty cube
    csdd = np.zeros([out_header['NAXIS3'],out_header['NAXIS2'],out_header['NAXIS1']])
    #fill it
    beam_in_pixels = [hdr['BMAJ']*3600./out_header['CDELT1'],hdr['BMIN']*3600./out_header['CDELT1']]

    x_proj = out_header['CRVAL1'] + (np.arange(out_header['NAXIS1'])+1 \
          - out_header['CRPIX1']) * out_header['CDELT1']
    y_proj = out_header['CRVAL2'] + (np.arange(out_header['NAXIS2'])+1 \
          - out_header['CRPIX2']) * out_header['CDELT2']
    zaxis = out_header['CRVAL3'] + (np.arange(out_header['NAXIS3'])+1 \
          - out_header['CRPIX3']) * out_header['CDELT3']
    x_map = np.resize(x_proj, [out_header['NAXIS2'],out_header['NAXIS1']])
    x_cube = np.resize(x_map, [out_header['NAXIS3'],out_header['NAXIS2'],out_header['NAXIS1']])

    y_map = np.transpose(np.resize(y_proj, [out_header['NAXIS2'],out_header['NAXIS1']]))
    y_cube = np.resize(y_map, [out_header['NAXIS3'],out_header['NAXIS2'],out_header['NAXIS1']])

    r_map = np.sqrt(x_cube**2+y_cube**2)
    r_map[r_map > np.nanmax(profiles[variables_we_need.index('RADI')])] = 0.
    #float('NaN')

    vel_cube=np.transpose(np.resize(zaxis,[out_header['NAXIS1'],out_header['NAXIS2'],len(zaxis)]),(2,1,0))/1000.

    Vp = vrot_r(r_map)*np.sin(np.radians(incl_r(r_map)))*np.cos(np.arctan(y_cube/x_cube))

    csdd = sbr_r(r_map)*np.exp(-0.5*((vel_cube-Vp)/sdis_r(r_map))**2)
    csdd[~np.isfinite(csdd)] = 0.
    csdd= ndimage.gaussian_filter(csdd,sigma= [0,beam_in_pixels[1],beam_in_pixels[0]],order=0.)
    noise = 1e-5

    csdd[ csdd < noise] = float('NaN')
    fits.writeto(input_parameters.csdd_cube, csdd,out_header,overwrite = True)

    mean_weight_r = np.nansum(r_map[8:,:,:]*csdd[8:,:,:], axis=1)/ np.nansum(csdd[8:,:,:], axis=1)
    ax = ['CDELT','CTYPE','CUNIT','CRPIX','CRVAL']
    for par in ax:
        try:
            out_header[f'{par}2'] = out_header[f'{par}3']
            out_header.remove(f'{par}3')
        except KeyError:
            pass
    out_header['CRPIX2'] = out_header['CRPIX2']-8
    fits.writeto('Weighted_R.fits', mean_weight_r, out_header,overwrite = True)



if __name__ == '__main__':
    createCSDDmaps(sys.argv[1:])
