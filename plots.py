


#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.stats import chi2
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import os
from netCDF4 import Dataset
import gsw 
from os import listdir
from os.path import isfile, join
from sklearn import linear_model
#import statsmodels.api as sm
from functions import get_hydro, throw_points, interp_to_edges, weights, nanrid, pdf, raw_pdf_plot, pdf_plot, remove_outliers, contour_plots
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

# plot PDFs
# now color dots by epsilon magnitude

figure_path = "/home/bryan/data/dlm/figures/"
data_path = "/home/bryan/data/dlm/data/"
scatter_path_SA_CT = figure_path + "scatter_SA_CT/"
filename = 'microdata.h5'

# get data
f = h5py.File( data_path + filename , 'r')
N2 = f['N2'][:]
CT = f['CT'][:]
SA = f['SA'][:]
eps = f['eps'][:]
z = f['z'][:]
Np = np.shape(eps)[0]

# *** raw data pdfs ***
#raw_pdf_plot( N2, SA, CT, eps )


# *** remove outliers ***
[N2, SA, CT, eps, z] = remove_outliers( N2, SA, CT, eps, z )


# *** pdfs and stats ***
#pdf_plot( N2, SA, CT, eps )
#[eps_mu,eps_sig,eps_sk,eps_fl,eps_min,eps_max] = pdf( eps )


# *** 2D contour plots ***
#contour_plots( N2, SA, CT, eps )


# *** 3D plots ***
print(type(CT))
zdata = np.log10(eps) 
ydata = CT 
xdata = SA 
#xdata,ydata,zdata = np.meshgrid(SA,CT,np.log10(eps))

for ii in range(64,360,1):
 fig = plt.figure()
 ax = Axes3D(fig)
 print(ii)
 ax.scatter3D(xdata, ydata, zdata, zdir='z', s=20, c=zdata, cmap='viridis',depthshade=False);
 #ax.contour3D(xdata, ydata, zdata, 100, cmap='viridis');
 ax.set_xlabel(r"S$_A$",fontsize=13)
 ax.set_ylabel(r"$\Theta$",fontsize=13)
 ax.view_init(elev=25., azim=ii)
 if ii < 10:
  plotname = scatter_path_SA_CT +'00%d.png' % ii
 elif ii < 100:
  plotname = scatter_path_SA_CT +'0%d.png' % ii
 elif ii < 1000:
  plotname = scatter_path_SA_CT +'%d.png' % ii
 plt.savefig(plotname,format="png"); plt.close(fig);
 fig = []
 ax = []



"""
plotname = figure_path +'scatter_eps_3d_N2_CT.png' 
fig = plt.figure()
ax = Axes3D(fig)
#zdata = eps/np.mean(eps)
#ydata = CT/np.mean(CT)
#xdata = SA/np.mean(SA)
zdata = np.log10(eps)
ydata = CT #/np.mean(CT)
xdata = N2 #/np.mean(SA)
#plt.axis('tight')
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis');
#ax.set_xlabel(r"S$_A/\overline{S}_A$",fontsize=13)
#ax.set_ylabel(r"$\Theta/\overline{\Theta}$",fontsize=13)
ax.set_xlabel(r"$N^2$",fontsize=13)
ax.set_ylabel(r"$\Theta$",fontsize=13)
#ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
ax.set_zlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
"""

