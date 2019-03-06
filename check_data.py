

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
import functions as fn #import get_hydro, throw_points, interp_to_edges, weights, nanrid


figure_path = "./figures/" #"/home/bryan/data/dlm/figures/"
output_path = "/home/bryan/data/dlm/data/"
micro_path = "/home/bryan/data/dlm/test/" #mislabeled_micro/"  #UCSD_microstructure/"

#N2 = np.empty([0])
#SA = np.empty([0])
#CT = np.empty([0])
#eps = np.empty([0])
#z = np.empty([0])

# get all folder names in UCSD micro
folder_names = [x[0] for x in os.walk(micro_path)]
Nfold = np.shape(folder_names)[0]
folder_names = folder_names[1:Nfold]

# loop over folder path
count = 0
plot_filenames = []

for i in range(0,Nfold-1):

 data_path = folder_names[i] 
 #print('folder =', data_path)

 onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
 Nfiles = np.shape(onlyfiles)[0] # number of files (time steps)

 for j in range(0,Nfiles):
   my_file = data_path + '/' + onlyfiles[j] # "bbtre96_10.nc"
   print('file =', my_file)
   f = Dataset(my_file, mode='r')
   print(f.variables)
   """
   [N2j, SAj, CTj, epsj, zj] = fn.get_hydro(my_file,count)
   N2=np.concatenate((N2,N2j),axis=0)
   SA=np.concatenate((SA,SAj),axis=0)
   CT=np.concatenate((CT,CTj),axis=0)
   eps=np.concatenate((eps,epsj),axis=0)
   z=np.concatenate((z,zj),axis=0)
   count = count + 1
   plot_filenames = np.append(plot_filenames,my_file)
   """

# clean up the data:
#[N2, SA, CT, eps, z] = fn.nanrid( N2, SA, CT, eps, z )
#[N2, SA, CT, eps, z] = fn.remove_outliers( N2, SA, CT, eps, z )



