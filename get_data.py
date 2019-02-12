# get_data.py
# Bryan Kaiser
# 1/30/2019

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
import statsmodels.api as sm
import functions as fn #import get_hydro, throw_points, interp_to_edges, weights, nanrid


figure_path = "./figures/" #"/home/bryan/data/dlm/figures/"
output_path = "/home/bryan/data/dlm/data/"
micro_path = "/home/bryan/data/dlm/UCSD_microstructure/"

N2 = np.empty([0])
SA = np.empty([0])
CT = np.empty([0])
eps = np.empty([0])
z = np.empty([0])

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
   [N2j, SAj, CTj, epsj, zj] = fn.get_hydro(my_file,count)
   N2=np.concatenate((N2,N2j),axis=0)
   SA=np.concatenate((SA,SAj),axis=0)
   CT=np.concatenate((CT,CTj),axis=0)
   eps=np.concatenate((eps,epsj),axis=0)
   z=np.concatenate((z,zj),axis=0)
   count = count + 1
   plot_filenames = np.append(plot_filenames,my_file)


# clean up the data:
[N2, SA, CT, eps, z] = fn.nanrid( N2, SA, CT, eps, z )
[N2, SA, CT, eps, z] = fn.remove_outliers( N2, SA, CT, eps, z )


# pdf plots:
fn.pdf_plot( N2, SA, CT, eps, z )
#[eps_mu,eps_sig,eps_sk,eps_fl,eps_min,eps_max] = fn.pdf( eps )


# scatter plots:

epsbar = ( np.log10(eps) - np.log10(np.nanmean(eps)) ) / np.log10(np.nanmean(eps))
N2bar = ( np.log10(N2) - np.log10(np.nanmean(N2)) ) / np.log10(np.nanmean(N2))
CTbar = ( (CT) - (np.nanmean(CT)) ) / (np.nanmean(CT))
SAbar = ( (SA) - (np.nanmean(SA)) ) / (np.nanmean(SA))

plotname = figure_path +"scatter_N2.png" #%(start_time,end_time)
fig = plt.figure()
plt.scatter(np.log10(eps),np.log10(N2),marker='o', color='blue',alpha=0.3)#,label="computed")
plt.xlabel(r"log$_{10}({\varepsilon})$",fontsize=13);
plt.ylabel(r"log$_{10}(N^2)$",fontsize=13); 
plt.grid(True)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'scatter_CT.png' #%(start_time,end_time)
fig = plt.figure()
plt.scatter(np.log10(eps),CT,marker='o', color='blue',alpha=0.3)#,label="computed")
plt.xlabel(r"log$_{10}({\varepsilon})$",fontsize=13);
plt.ylabel(r"$\overline{\Theta}$",fontsize=13); 
plt.grid(True)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'scatter_SA.png' #%(start_time,end_time)
fig = plt.figure()
plt.scatter(np.log10(eps),SA,marker='o', color='blue',alpha=0.3)#,label="computed")
plt.xlabel(r"log$_{10}({\varepsilon})$",fontsize=13);
plt.ylabel(r"$\overline{S}_A$",fontsize=13); 
plt.grid(True)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'scatter_z.png' #%(start_time,end_time)
fig = plt.figure()
plt.scatter(np.log10(eps),z,marker='o', color='blue',alpha=0.3)#,label="computed")
plt.xlabel(r"log$_{10}(\overline{\varepsilon})$",fontsize=13);
plt.ylabel(r"$z$",fontsize=13); 
plt.grid(True)
plt.savefig(plotname,format="png"); plt.close(fig);

nu = 1e-6
Reb = eps/(nu*abs(N2))
plotname = figure_path +'scatter_Reb.png' #%(start_time,end_time)
fig = plt.figure()
plt.scatter(np.log10(eps),np.log10(Reb),marker='o', color='blue',alpha=0.3)#,label="computed")
plt.xlabel(r"log$_{10}(\overline{\varepsilon})$",fontsize=13);
plt.ylabel(r"log$_{10}$Re$_b$",fontsize=13); 
#plt.axis([-10.5,-4.,-2000.,500000.])
plt.grid(True)
plt.savefig(plotname,format="png"); plt.close(fig);


# write names to .txt file
with open(output_path + "plotnames.txt", "w") as text_file:
 for  k in range(0,count):
  variable = str(k)
  text_file.write( variable + '  ' + plot_filenames[k] + '\n')

# write data to .h5 file
h5_filename = output_path + "microdata.h5" 
f2 = h5py.File(h5_filename, "w")
dset = f2.create_dataset('N2', data=N2, dtype='f8')
dset = f2.create_dataset('SA', data=SA, dtype='f8')
dset = f2.create_dataset('CT', data=CT, dtype='f8')
dset = f2.create_dataset('eps', data=eps, dtype='f8')
dset = f2.create_dataset('z', data=z, dtype='f8')

print('\nDone!\n')

