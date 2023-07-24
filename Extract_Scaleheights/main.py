#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-

#This program calculates the scale height of an edge-on galaxy following the procedures laid out in Olling 1995 and O'Brien 2007
# For now this only works for edge-on galaxies.
from Extract_Scaleheights.functions import setup_input_parameters,createCSDDmaps,\
     measure_FWHM
import pk_common_functions.functions as cf
import sys
import numpy as np
import warnings
import scipy.stats as stats
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    #matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Ellipse
    import matplotlib.axes as maxes
def main(argv):
    # Setup the input parameters
    input_parameters = setup_input_parameters(argv)
    #First we create our weighted R_map for the mapping purposes

    r_max = createCSDDmaps(deffile_name = input_parameters.deffile, \
        cube_name = input_parameters.cube_name, \
        weighted_map_name= input_parameters.weighted_r)

    # Then we need some info about the galaxy to not over do the fitting
    variables_we_need = ['INCL','INCL_2','PA','PA_2','VROT','VROT_2',\
        'RMS','BMAJ','BMIN','VSYS','XPOS','YPOS','Z0','RADI','NDISKS']
    #First we the tirific PARAMETERS
    profiles =  cf.load_tirific(input_parameters.deffile,\
        Variables = variables_we_need,dict=True,array=True)
    if profiles['NDISKS'][0] == 1:
        for var in ['INCL','PA','VROT']:
            profiles[f'{var}_2'] = profiles[var]

    pa = np.mean([profiles['PA'][0:2],profiles['PA_2'][0:2]])
    vel_in_cube = np.max([profiles['VROT'],profiles['VROT_2']])\
                *np.sin(np.radians(np.max([profiles['INCL'],profiles['INCL_2']])))
    vel_range = 1.1*vel_in_cube
    beam =  [profiles['BMAJ'][0],profiles['BMIN'][0]]
    print(f'The size of the beam is {beam}')
    vals = measure_FWHM(cube_name = input_parameters.cube_name, beam= beam, \
        center = [profiles['XPOS'][0],profiles['YPOS'][0],profiles['VSYS'][0]]\
        ,r_max = r_max, pa = pa , map_name = input_parameters.weighted_r, \
        vel_range = vel_range , noise= profiles['RMS'][0]\
        ,rotated_cube_name = input_parameters.rotated_cube)
    print(vals[:,0])
    if float(input_parameters.distance) != -1:
        tmp1 = cf.convertskyangle(vals[:,0],\
            distance=float(input_parameters.distance))
        tmp2 = cf.convertskyangle(vals[:,1]/(2*(np.sqrt(2*np.log(2)))),\
            distance=float(input_parameters.distance))
        vals[:,0] = tmp1
        vals[:,1] = tmp2
        bin_size= cf.convertskyangle(beam[0],distance=float(input_parameters.distance))
    else:
        bin_size=beam[0]
    #Make a scater plot of all the value
    plt.scatter(vals[:,0],vals[:,1],marker='o')

    #Then bin them all starting from the minimum radius upto the maximum
    radius_bins = np.linspace(np.nanmin(vals[:,0]),np.nanmax(vals[:,0]), \
        int((np.nanmax(vals[:,0])-np.nanmin(vals[:,0]))/bin_size)+1)

    bin_means,bin_edges,bin_num = stats.binned_statistic(vals[:,0],vals[:,1],\
        bins=radius_bins)
    bin_errors,bin_edges,bin_num = stats.binned_statistic(vals[:,0],vals[:,1],\
        bins=radius_bins,statistic='std')

    plt.scatter([(x-y)/2.+y for x,y in zip(bin_edges[1:],bin_edges)],bin_means,marker='*',color='r')
    plt.errorbar([(x-y)/2.+y for x,y in zip(bin_edges[1:],bin_edges)],bin_means,yerr=bin_errors,fmt = 'none', ecolor='red')
    if float(input_parameters.distance) != -1:
        plt.plot(cf.convertskyangle(profiles['RADI'],distance=float(input_parameters.distance)),\
                cf.convertskyangle(profiles['Z0']*(2.0*np.sqrt(2.0*np.log(2.0))),\
                distance=float(input_parameters.distance)),'k')
        plt.xlabel('Galactocentric Radius (kpc)')
        plt.ylabel(f'Gaussian Dispersion (kpc)')
    else:
        plt.plot(profiles['RADI'],profiles['Z0']*(2.0*np.sqrt(2.0*np.log(2.0))),'k')
        plt.xlabel('Galactocentric Radius (")')
        plt.ylabel(f'Gaussian Dispersion (")')
    limits = [np.nanmin(bin_means-bin_errors),np.nanmax(bin_means+bin_errors)]

    plt.ylim(*limits)
    plt.savefig(input_parameters.plot_name)
if __name__ == '__main__':
    main(sys.argv[1:])
