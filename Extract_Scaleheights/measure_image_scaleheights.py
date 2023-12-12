#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-

# This programs measures the scale height from an image
# For now this only works for edge-on galaxies.


# import python packages
import sys
import os
import copy
import numpy as np
import scipy.interpolate as spirp
import scipy.stats as stats
from optparse import OptionParser
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import common_functions as cf
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    #matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Ellipse
    import matplotlib.axes as maxes
image_in= '/home/peter/Misc/Vertical Profiles/ESO270-G017.W1.clean.fits'
image_out = '/home/peter/Misc/Vertical Profiles/ESO270-G017.W1.clean_rotated.fits'
Distance = 6.95
PA =105.7
center = ['13:34:47.3','-45:32:51']
diameter = 80.
binsize = 5.
height =  60.

no_bins = int(diameter/2./binsize)

image = fits.open(image_in)
im_coord = WCS(image[0].header)
center_deg = cf.convertRADEC(*center, invert=True, colon=True)
center_pix = im_coord.wcs_world2pix(*center_deg,1)
pixel_size = np.mean(proj_plane_pixel_scales(im_coord)*3600.)
image_rotated = cf.rotateImage(image[0].data, PA-90, center_pix)
fits.writeto(image_out,image_rotated,image[0].header,overwrite = True)

binsize_in_pixel = int(binsize/pixel_size)
radius_in_pixel = int(diameter/2./pixel_size)
height_in_pixel = int(height/pixel_size)

found_values ={'cube_type': ['Restored','Restored','Model', 'Unmasked Model' ,'Residual','Model+Residual','Unmasked+Residual']}
labelfont= {'family':'Times New Roman',
            'weight':'normal',
            'size':18}
plt.rc('font',**labelfont)
main = plt.figure(1,figsize=(18.,6.), dpi=100, facecolor='w', edgecolor='k')
gs = main.add_gridspec(6,10)
ax= main.add_subplot(gs[1:6,0:4])
ax.set_xlabel('Minor Axis offset (arcsec)')
ax.title.set_text('Negative major axis bins')
ax.set_ylabel('Intensity')

ax2= main.add_subplot(gs[1:6,6:])
ax2.set_xlabel('Minor Axis offset (arcsec)')
ax2.title.set_text('Positive major axis bins')
ax2.set_ylabel('Intensity')
yaxis = np.array(range(height_in_pixel*2),dtype=float)-height_in_pixel
h_s= []
for i in range(no_bins):
    print(i)
    neg_profile = np.mean(image_rotated[int(center_pix[1])-height_in_pixel:int(center_pix[1])+height_in_pixel,
                            int(center_pix[0])-(i+1)*binsize_in_pixel:int(center_pix[0])-(i)*binsize_in_pixel],axis=1) - \
                            np.mean([np.mean(image_rotated[int(center_pix[1])+height_in_pixel:int(center_pix[1])+2*height_in_pixel,
                            int(center_pix[0])-(i+1)*binsize_in_pixel:int(center_pix[0])-(i)*binsize_in_pixel]),\
                            np.mean(image_rotated[int(center_pix[1])-2*height_in_pixel:int(center_pix[1])-height_in_pixel,
                            int(center_pix[0])-(i+1)*binsize_in_pixel:int(center_pix[0])-(i)*binsize_in_pixel])])
    peak1,center1,exp1 = cf.fit_exponential(yaxis[yaxis >=0. ],neg_profile[yaxis >=0.] )
    if abs(exp1) > height_in_pixel:
        exp1 = float('NaN')
    peak2,center2,exp2 = cf.fit_exponential(abs(yaxis[yaxis <=0. ]),neg_profile[yaxis <=0.] )
    if abs(exp2) > height_in_pixel:
        exp2 = float('NaN')
    p = ax.plot(yaxis*pixel_size,neg_profile)
    color = p[-1].get_color()
    if not np.isnan(exp1):
        ax.plot(yaxis[yaxis >=0. ]*pixel_size,cf.exponential_function(yaxis[yaxis >=0. ],peak1,center1,exp1 ),'--',color=color)

    if not np.isnan(exp2):
        ax.plot(yaxis[yaxis <=0. ]*pixel_size,cf.exponential_function(abs(yaxis[yaxis <=0. ]),peak2,center2,exp2 ),'--',color=color)

    if not np.isnan(exp2) and not np.isnan(exp1):
        print(exp2*pixel_size,exp1*pixel_size)
        h_s.append(np.nanmean([exp1,abs(exp2)])*pixel_size)
    elif not np.isnan(exp2):
        h_s.append(abs(exp2)*pixel_size)
    elif not np.isnan(exp1):
        h_s.append(exp1*pixel_size)

    pos_profile = np.mean(image_rotated[int(center_pix[1])-height_in_pixel:int(center_pix[1])+height_in_pixel,
                            int(center_pix[0])+(i)*binsize_in_pixel:int(center_pix[0])+(i+1)*binsize_in_pixel],axis=1) - \
                            np.mean([np.mean(image_rotated[int(center_pix[1])+height_in_pixel:int(center_pix[1])+2*height_in_pixel,
                            int(center_pix[0])+(i)*binsize_in_pixel:int(center_pix[0])+(i+1)*binsize_in_pixel]),\
                            np.mean(image_rotated[int(center_pix[1])-2*height_in_pixel:int(center_pix[1])-height_in_pixel,
                            int(center_pix[0])+(i)*binsize_in_pixel:int(center_pix[0])+(i+1)*binsize_in_pixel])
                            ])
    peak1,center1,exp1 = cf.fit_exponential(yaxis[yaxis >=0. ],pos_profile[yaxis >=0.] )
    if abs(exp1) > height_in_pixel:
        exp1 = float('NaN')
    peak2,center2,exp2 = cf.fit_exponential(abs(yaxis[yaxis <=0. ]),pos_profile[yaxis <=0.] )
    if abs(exp2) > height_in_pixel:
        exp2 = float('NaN')
    p = ax2.plot(yaxis*pixel_size,pos_profile, color=color)
    if not np.isnan(exp1):
        ax2.plot(yaxis[yaxis >=0. ]*pixel_size,cf.exponential_function(yaxis[yaxis >=0. ],peak1,center1,exp1 ),'--',color=color)
    if not np.isnan(exp2):
        ax2.plot(yaxis[yaxis <=0. ]*pixel_size,cf.exponential_function(abs(yaxis[yaxis <=0. ]),peak2,center2,exp2 ),'--',color=color)

    if not np.isnan(exp2) and not np.isnan(exp1):
        print(exp2*pixel_size,exp1*pixel_size)
        h_s.append(np.nanmean([exp1,abs(exp2)])*pixel_size)
    elif not np.isnan(exp2):
        h_s.append(abs(exp2)*pixel_size)
    elif not np.isnan(exp1):
        h_s.append(exp1*pixel_size)

plt.savefig('/home/peter/Misc/Vertical Profiles/scale_heights.png')
plt.close()
print(h_s)
print(f"The scale height is {np.mean(h_s)} arcsec and {cf.convertskyangle(np.mean(h_s),distance=Distance)} kpc")
