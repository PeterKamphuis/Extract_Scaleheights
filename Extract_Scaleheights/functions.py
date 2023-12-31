#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-

#This program calculates the scale height of an edge-on galaxy following the procedures laid out in Olling 1995 and O'Brien 2007
# For now this only works for edge-on galaxies.


# import python packages
import Extract_Scaleheights
import sys
import os
import copy
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import scipy.stats as stats
from optparse import OptionParser
from astropy.io import fits
from astropy.wcs import WCS
from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
from typing import List, Optional

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    #matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Ellipse
    import matplotlib.axes as maxes

# Then let's add our main directory
import pk_common_functions.functions as cf
import psutil


# A dataclass object for Omega Conf to use for input
@dataclass
class defaults:
    configuration_file: Optional[str] = None
    ncpu: int = len(psutil.Process().cpu_affinity())
    cube_name: Optional[str] = None #'Define the cube to be analysed.'
    deffile: Optional[str] = None # Input Tirific Def File
    weighted_r: str = 'r_map.fits' #Name of the output CSDD cube
    rotated_cube: str = 'CSDD_Rotated_Cube.fits' #'Name of the output rotated cube
    plot_name: str = f'scale_height_plot.png' #  'Name of the output scaleheight plot in FWHM
    distance: float = 1. #Distance to the galaxy
    print_examples: bool = False

class InputError(Exception):
    pass

def setup_header(cube_name):
#open our  cube as a template
    hdr = fits.getheader(cube_name)
    if hdr['CDELT3'] < 100.:
        hdr['CDELT3'] = hdr['CDELT3']*1000.
        hdr['CRVAL3'] = hdr['CRVAL3']*1000.
        hdr['CUNIT3'] = 'M/S'
    hdr = cf.reduce_header_axes(hdr)
    # make an empty fake cube centered on the galaxy
    out_header = copy.deepcopy(hdr)
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
    return out_header
setup_header.__doc__ = '''
Setup the header to use for the CSDD cube
'''

def create_rmap(hdr,radi):
    x_proj = hdr['CRVAL1'] + (np.arange(hdr['NAXIS1'])+1 \
          - hdr['CRPIX1']) * hdr['CDELT1']
    y_proj = hdr['CRVAL2'] + (np.arange(hdr['NAXIS2'])+1 \
          - hdr['CRPIX2']) * hdr['CDELT2']
    x_map = np.resize(x_proj, [hdr['NAXIS2'], hdr['NAXIS1']])
    x_cube = np.resize(x_map, [hdr['NAXIS3'], hdr['NAXIS2'], hdr['NAXIS1']])

    y_map = np.transpose(np.resize(y_proj, [hdr['NAXIS2'], hdr['NAXIS1']]))
    y_cube = np.resize(y_map, [hdr['NAXIS3'], hdr['NAXIS2'], hdr['NAXIS1']])

    r_map = np.sqrt(x_cube**2+y_cube**2)
    r_max = np.nanmax(radi)
    r_map[r_map > r_max] = 0.
    return r_map,r_max,x_cube,y_cube

# This program makes CSDD maps as described in Olling 1995
def createCSDDmaps(deffile_name = 'Input.def', cube_name = 'Default_Cube.fits', weighted_map_name= 'R_Mapping.fits'):
    #Constants that are used in the code
    H_0 = 70. #km/s/Mpc #Hubble constant
    #The variables we need from the tirific model
    variables_we_need = ['RADI','INCL','INCL_2','PA','PA_2','VROT','VROT_2',\
        'SBR','SBR_2','NDISKS','SDIS','SDIS_2','VSYS','XPOS','YPOS']
    #And we extract them
    profiles =  cf.load_tirific(deffile_name,Variables = variables_we_need,\
        dict=True,array=True)
    #Make sure that everything is filles also for single disks
    variables = ['INCL','PA','VROT','SBR','SDIS']
    #Create symmetric functions
    symmetric = {}
    for var in variables:
        if float(profiles['NDISKS'][0]) == 1:
            symmetric[var] = profiles[var]
        else:
            symmetric[var]= interpolate.interp1d(profiles['RADI'],\
                [np.mean([x,y]) for x,y in zip(profiles[var],\
                profiles[f'{var}_2'])])

    csdd_hdr = setup_header(cube_name)
    #Create rmap
    r_map,r_max,x_cube,y_cube = create_rmap(csdd_hdr,profiles['RADI'])

    # Set up a WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csdd_wcs = WCS(csdd_hdr)
    #make an empty cube
    csdd = np.zeros([csdd_hdr['NAXIS3'],csdd_hdr['NAXIS2'],csdd_hdr['NAXIS1']])

    zaxis = csdd_hdr['CRVAL3'] + (np.arange(csdd_hdr['NAXIS3'])+1 \
          - csdd_hdr['CRPIX3']) * csdd_hdr['CDELT3']

    vel_cube=np.transpose(np.resize(zaxis,[csdd_hdr['NAXIS1'],csdd_hdr['NAXIS2'],\
        len(zaxis)]),(2,1,0))/1000.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Vp = symmetric['VROT'](r_map)*np.sin(np.radians(symmetric['INCL'](r_map)))\
            *np.cos(np.arctan(y_cube/x_cube))

    csdd = symmetric['SBR'](r_map)*np.exp(-0.5*((vel_cube-Vp)/symmetric['SDIS'](r_map))**2)
    csdd[~np.isfinite(csdd)] = 0.

    beam_in_pixels = [csdd_hdr['BMAJ']*3600./csdd_hdr['CDELT1'],\
        csdd_hdr['BMIN']*3600./csdd_hdr['CDELT1']]

    csdd= ndimage.gaussian_filter(csdd,sigma= [0,beam_in_pixels[1],beam_in_pixels[0]],order=0.)
    noise = 1e-5
    fits.writeto('Test_R_Cube.fits',r_map, csdd_hdr,overwrite = True)
    fits.writeto('CSDD_Cube.fits',csdd, csdd_hdr,overwrite = True)

    csdd[ csdd < noise] = float('NaN')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_weight_r = np.nansum(r_map[:,:,:]*csdd[:,:,:], axis=1)/\
            np.nansum(csdd[:,:,:], axis=1)

    for i in range(len(mean_weight_r[0,:])):
        if np.isnan(mean_weight_r[0,i]) or (mean_weight_r[0,i] == 0.):
            last_value= i-1
            break
    for i in range(len(mean_weight_r[:,0])):
        if np.isnan(mean_weight_r[i,last_value]) or \
            (mean_weight_r[i,last_value] == 0.):
            cut = i
            break
    mean_weight_r = mean_weight_r[cut:,:]


            #break



    ax = ['CDELT','CTYPE','CUNIT','CRPIX','CRVAL']
    for par in ax:
        try:
            csdd_hdr[f'{par}2'] = csdd_hdr[f'{par}3']
            csdd_hdr.remove(f'{par}3')
        except KeyError:
            pass
    csdd_hdr['CRPIX2'] = csdd_hdr['CRPIX2']-cut
    csdd_hdr['CDELT1'] = 1.
    csdd_hdr['CUNIT1'] = 'pixels'
    csdd_hdr['CUNIT2'] = 'channels'
    csdd_hdr['CTYPE2'] = 'channel offset'
    csdd_hdr['CDELT2'] = 1.
    csdd_hdr['CRVAL2'] = 0.
    csdd_hdr['BUNIT'] = 'Radius'
    fits.writeto(weighted_map_name, mean_weight_r, csdd_hdr,overwrite = True)

    return r_max


def measure_FWHM(cube_name = 'Input_Cube.fits', center = [0,0,0] ,\
                r_max = 100.,pa =90. ,map_name = 'r_map.fits', beam = [1.,1.],
                vel_range = [0.,0.] , noise= 0.,rotated_cube_name = ''):
    cube = fits.open(cube_name)
    hdr = cf.reduce_header_axes(cube[0].header)
    data = cf.reduce_data_axes(cube[0].data)

    if hdr['CDELT3'] < 100.:
        hdr['CDELT3'] = hdr['CDELT3']*1000.
        hdr['CRVAL3'] = hdr['CRVAL3']*1000.
        hdr['CUNIT3'] = 'M/S'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube_wcs = WCS(hdr)
    # get the central pixels
    x_center,y_center,z_center = cube_wcs.wcs_world2pix(center[0],center[1],\
        (center[2])*1000.,0.)
    print(f'This is the central channel {z_center}')
    # PA is the angle to the receding sid from north, we always want the receding side to be negative offsets
    # Pa runs counter clockwise but we are rotating clockwise so pa-90
    rotated_cube = cf.rotateCube(data, pa-90, [x_center,y_center])
    # We blank everything in the cube that is less than 3 * the noise value
    rotated_cube[rotated_cube < 3.*noise] = float('NaN')
    if rotated_cube_name != '':
        fits.writeto(rotated_cube_name, rotated_cube, hdr,overwrite = True)
    z_pix= int(z_center)
    # our maximum r in pixels =
    pix_size = np.mean([abs(hdr['CDELT1']),abs(hdr['CDELT2'])])*3600.
    r_max_pix = r_max/pix_size
    #
    weight_map = fits.open(map_name)
    weight_xaxis = weight_map[0].header['CRVAL1'] + (np.arange(weight_map[0].header['NAXIS1'])+1 \
              - weight_map[0].header['CRPIX1']) * weight_map[0].header['CDELT1']
    weight_zaxis= weight_map[0].header['CRVAL2'] + (np.arange(weight_map[0].header['NAXIS2'])+1 \
              - weight_map[0].header['CRPIX2']) * weight_map[0].header['CDELT2']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_min_pix = np.nanmin(weight_map[0].data[weight_map[0].data > 0.])/pix_size
        if r_min_pix < 2:
            r_min_pix=2
        print(r_min_pix,np.nanmin(weight_map[0].data[weight_map[0].data > 0.]) )
    # Get the minimum and maximum velocity channel to analyse
    # get the central pixels
    x_tmp,y_tmp,z_range = cube_wcs.wcs_world2pix(center[0],center[1],\
        (center[2]+vel_range)*1000.,0.)
    x_tmp,y_tmp,z_start = cube_wcs.wcs_world2pix(center[0],center[1],\
        (center[2]-vel_range)*1000.,0.)


    yaxis = np.arange(hdr['NAXIS2'])
    # run through the relevant channels
    found_values = []
    succes_fit=0
    failed_fit=0
    no_fit= 0
    for i in range(int(z_start)-2,int(z_range)+2):
        print(f'Processing Channel {i}')
        if int(z_center+i) < hdr['NAXIS3'] and int(z_center-i) >= 0:
            #run trhough the relevant x_range ranges
            for x in range(int(r_min_pix)-2,int(r_max_pix)+2):
                #x is the same in the map as in offset no need to map it
                #for z the i value corresponds to what it is in the axis
                map_z = np.where(i == weight_zaxis)[0]
                if len(map_z) > 0:
                    map_z = map_z[0]
                else:
                    map_z = -1
                # is our mapping correct?
                #print(f'The input is {i} {x}')
                #print(f'Ourmapping is {map_z} {x}')
                if 0 < x < weight_map[0].header['NAXIS1'] and \
                    0 < map_z < weight_map[0].header['NAXIS2']:
                    R = weight_map[0].data[map_z,x]
                    #print(f'And then R is {R}')
                    #extract the profile and fit
                    if np.isfinite(R):
                        channels = np.array([rotated_cube[int(z_center-i),:,:],\
                            rotated_cube[int(z_center+i),:,:]])
                        #print(channels.shape)
                        #print(f'extracting channels at {z_pix-i} and {z_pix+i} with i= {i} and center {z_center}')
                        for j in [0,1]:
                            no_fit += 1
                            #receding side is negative offsets so higher channel are at min x
                            if j == 0:
                                offset= 1
                            else:
                                offset = -1.
                            #print(f'extracting the profile at {int(x_center+offset*x)} for x= {x} and the center = {x_center}')
                            profile = channels[j,:,int(x_center+offset*x)]
                            #print(profile)
                            #if i >11:
                            #    exit()
                            if np.nansum(profile) > 0. and np.nanmax(profile) \
                                > 5.*noise:
                                    #fit a gausssin
                                fit_axis = yaxis[np.isfinite(profile)]
                                diff = np.array([float(x-y) for x,y \
                                    in zip(fit_axis[1:],fit_axis)],dtype=float)
                                trig=False
                                while not all([f == 1 for f in diff]):
                                    jump_location = np.where(diff > 1.)[0]
                                    if len(jump_location) > 1:
                                        cont_size = np.array([float(x-y) for\
                                            x,y in zip(jump_location[1:],\
                                            jump_location)],dtype=float)
                                        cont_size = np.hstack((jump_location[0]+1\
                                            ,cont_size,len(fit_axis)-jump_location[-1]-1))
                                        big = np.where(cont_size == \
                                            np.nanmax(cont_size))[0]
                                            #print(big)
                                        if big[0] == 0:
                                            fit_axis =fit_axis[:int(jump_location[big[0]])+1]
                                        else:
                                            fit_axis =fit_axis[jump_location[big[0]-1]\
                                                +1:int(jump_location[big[0]-1]+\
                                                np.nanmax(cont_size)+1)]
                                    else:
                                        if len(diff)-jump_location[0] > jump_location[0]:
                                            fit_axis = fit_axis[jump_location[0]+1:]
                                        else:
                                            fit_axis = fit_axis[:jump_location[0]+1]
                                    diff = np.array([float(x-y) for x,y in \
                                        zip(fit_axis[1:],fit_axis)],dtype=float)

                                fit_profile = profile[fit_axis]

                                #if not continuous we need to eak out the peak

                                if len(fit_axis) > 3.:
                                    parms = cf.fit_gaussian(fit_axis,fit_profile)

                                    if np.sqrt((parms[2]*(2.0*np.sqrt(2.0*\
                                        np.log(2.0)))*pix_size)**2-beam[0]**2) \
                                            > 1000.:
                                        plt.plot(xaxis[np.isfinite(profile)], profile[np.isfinite(profile)])
                                        plt.plot(fit_axis, fit_profile,'-r')
                                        plt.plot(xaxis,cf.gaussian_function(xaxis, *parms))
                                        plt.plot([y_center-parms[2]*(np.sqrt(2.0*np.log(2.0))),y_center+parms[2]*(np.sqrt(2.0*np.log(2.0)))],\
                                            [np.max(profile[np.isfinite(profile)])/2.,np.max(profile[np.isfinite(profile)])/2.])
                                        plt.show()

                                    if np.isfinite(parms[0]) and parms[0] > \
                                        5.*noise and (parms[2]*\
                                        (2.0*np.sqrt(2.0*np.log(2.0)))*pix_size)**2\
                                        -beam[0]**2 > 0.:
                                        found_values.append([R*offset,\
                                            np.sqrt((parms[2]*(2.0*np.sqrt(2.0*\
                                            np.log(2.0)))*pix_size)**2-beam[0]**2)])
                                        succes_fit += 1
                                    else:
                                        failed_fit += 1
                                        print(f'The fit for {R*offset} failed')
                else:
                    pass
    print(f'Out of {no_fit} fit {succes_fit} were succesful and {failed_fit} failed.')
    return np.array(found_values,dtype=float)

def setup_input_parameters(argv):

    if '-v' in argv or '--version' in argv:
        print(f"This is version {Extract_Scaleheights.__version__} of the program.")
        sys.exit()

    if '-h' in argv or '--help' in argv:
        print('''
Use Extract_Scaleheights in this way:
Create_CSDD configuration_file=inputfile.yml   where inputfile is a yaml config file with the desired input settings.
Create_CSDD -h print this message
Create_CSDD print_examples=True prints a yaml file (defaults.yml) with the default setting in the current working directory.
in this file values designated ??? indicated values without defaults.

All config parameters can be set directly from the command line by setting the correct parameters, e.g:
Create_CSDD cube_name=cube.fits distance=8.9
''')
        sys.exit()

    cfg = OmegaConf.structured(defaults)
    inputconf = OmegaConf.from_cli(argv)
    cfg_input = OmegaConf.merge(cfg,inputconf)
        # read command line arguments anything list input should be set in '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'

    if cfg_input.print_examples:
        with open('default.yml','w') as default_write:
            default_write.write(OmegaConf.to_yaml(cfg))
        sys.exit()

    if cfg_input.configuration_file:
        succes = False
        while not succes:
            try:
                yaml_config = OmegaConf.load(cfg_input.configuration_file)
        #merge yml file with defaults
                cfg = OmegaConf.merge(cfg,yaml_config)
                succes = True
            except FileNotFoundError:
                raise InputError(f'We could not find the file {cfg_input.configuration_file}')

    input_parameters = OmegaConf.merge(cfg,inputconf)

    if input_parameters.cube_name == None:
        raise InputError(f'There is no default for the cube_name.')
    if input_parameters.deffile == None:
        raise InputError(f'There is no default for the TiRiFiC file.')
    return input_parameters

setup_input_parameters.__doc__ =f'''
This function sets up the input parameters object through Omega Conf
'''


def main(argv):
    input_parameters = setup_input_parameters(argv)

    print(input_parameters)
    sys.exit()

    #First we create our weighted R_map for the mapping purposes
    r_max = createCSDDmaps(deffile_name = input_parameters.deffile, \
        cube_name = input_parameters.cube_name, \
        weighted_map_name= input_parameters.weighted_r)
    # Then we need some info about the galaxy to not over do the fitting
    variables_we_need = ['INCL','INCL_2','PA','PA_2','VROT','VROT_2','RMS','BMAJ','BMIN','VSYS','XPOS','YPOS','Z0','RADI']
    #First we the tirific PARAMETERS
    profiles =  cf.load_tirific(input_parameters.deffile,Variables = variables_we_need)
    pa = np.mean([profiles[variables_we_need.index('PA')][0:2],profiles[variables_we_need.index('PA_2')][0:2]])
    vel_in_cube = np.max([profiles[variables_we_need.index('VROT')],profiles[variables_we_need.index('VROT_2')]])\
                *np.sin(np.radians(np.max([profiles[variables_we_need.index('INCL')],profiles[variables_we_need.index('INCL_2')]])))
    vel_range = 1.1*vel_in_cube
    beam =  [profiles[variables_we_need.index('BMAJ')][0],profiles[variables_we_need.index('BMIN')][0]]
    vals = measure_FWHM(cube_name = input_parameters.cube_name, beam= beam, \
        center = [profiles[variables_we_need.index('XPOS')][0],\
        profiles[variables_we_need.index('YPOS')][0],\
        profiles[variables_we_need.index('VSYS')][0]],r_max = r_max,pa =pa ,\
        map_name = input_parameters.weighted_r, vel_range = vel_range ,\
        noise= profiles[variables_we_need.index('RMS')][0]\
        ,rot_cube_name = input_parameters.rotated_cube)

    if float(input_parameters.distance) != -1:
        tmp1 = cf.convertskyangle(vals[:,0],distance=float(input_parameters.distance))
        tmp2 = cf.convertskyangle(vals[:,1],distance=float(input_parameters.distance))
        vals[:,0] = tmp1
        vals[:,1] = tmp2
        bin_size= cf.convertskyangle(beam[0],distance=float(input_parameters.distance))
    else:
        bin_size=beam[0]
    #Make a scater plot of all the value
    plt.scatter(vals[:,0],vals[:,1],marker='o')

    #Then bin them all starting from the minimum radius upto the maximum
    radius_bins = np.linspace(np.nanmin(vals[:,0]),np.nanmax(vals[:,0]), int((np.nanmax(vals[:,0])-np.nanmin(vals[:,0]))/bin_size)+1)
    print(radius_bins)
    bin_means,bin_edges,bin_num = stats.binned_statistic(vals[:,0],vals[:,1],bins=radius_bins)
    bin_errors,bin_edges,bin_num = stats.binned_statistic(vals[:,0],vals[:,1],bins=radius_bins,statistic='std')
    print(bin_edges)
    print(len(bin_means),len([(x-y)/2.+y for x,y in zip(bin_edges[1:],bin_edges)]))

    plt.scatter([(x-y)/2.+y for x,y in zip(bin_edges[1:],bin_edges)],bin_means,marker='*',color='r')
    plt.errorbar([(x-y)/2.+y for x,y in zip(bin_edges[1:],bin_edges)],bin_means,yerr=bin_errors,fmt = 'none', ecolor='red')
    if float(input_parameters.distance) != -1:
        plt.plot(cf.convertskyangle(profiles[variables_we_need.index('RADI')],distance=float(input_parameters.distance)),\
                cf.convertskyangle(profiles[variables_we_need.index('Z0')]*(2.0*np.sqrt(2.0*np.log(2.0))),\
                distance=float(input_parameters.distance)),'k')
        plt.xlabel('Galactocentric Radius (kpc)')
        plt.ylabel(f'FWHM (kpc)')
    else:
        plt.plot(profiles[variables_we_need.index('RADI')],profiles[variables_we_need.index('Z0')]*(2.0*np.sqrt(2.0*np.log(2.0))),'k')
        plt.xlabel('Galactocentric Radius (")')
        plt.ylabel(f'FWHM (")')
    limits = [np.min(bin_means-bin_errors),np.max(bin_means+bin_errors)]
    plt.ylim(*limits)
    plt.savefig(input_parameters.final_plot)
if __name__ == '__main__':
    main(sys.argv[1:])
