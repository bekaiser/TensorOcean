# 

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

figure_path = "./figures/"

# =============================================================================    
# classes and functions

def interp_to_edges( self_center , ze , zc , flag ):
 # self is centers
 Ne = np.shape(ze)[0] # edge number
 Nc = np.shape(zc)[0] # center number
 # e1 c1 e2 c2 e3 c3 e4 c4 e5 c5 e6 c6 e7
 #    in  o in o  in o  in  o in o  in
 self_edge = np.zeros([Nc-1])
 
 if flag == 6:
   for j in range(2,Nc-3):
     c = np.transpose(weights(ze[j],zc[j-2:j+4],0))
     self_edge[j] = np.dot( c[0,:] , self_center[j-2:j+4] )
   c = np.transpose(weights(ze[1],zc[0:6],0))
   self_edge[0] = np.dot( c[0,:] , self_center[0:6] )
   c = np.transpose(weights(ze[2],zc[0:6],0))
   self_edge[1] = np.dot( c[0,:] , self_center[0:6] )
   c = np.transpose(weights(ze[Ne-1],zc[Ne-7:Ne-1],0))
   self_edge[Nc-2] = np.dot( c[0,:] , self_center[Ne-7:Ne-1] )
   c = np.transpose(weights(ze[Ne-2],zc[Ne-7:Ne-1],0))
   self_edge[Nc-3] = np.dot( c[0,:] , self_center[Ne-7:Ne-1] )

 if flag == 4:
   for j in range(1,Nc-2):
     c = np.transpose(weights(ze[j],zc[j-1:j+3],0))
     self_edge[j] = np.dot( c[0,:] , self_center[j-1:j+3] )
   c = np.transpose(weights(ze[1],zc[0:4],0))
   self_edge[0] = np.dot( c[0,:] , self_center[0:4] )
   c = np.transpose(weights(ze[Ne-1],zc[Ne-5:Ne-1],0))
   self_edge[Nc-2] = np.dot( c[0,:] , self_center[Ne-5:Ne-1] )

 if flag == 2:
   for j in range(1,Nc-2):
     c = np.transpose(weights(ze[j],zc[j:j+2],0))
     self_edge[j] = np.dot( c[0,:] , self_center[j:j+2] )
   c = np.transpose(weights(ze[1],zc[0:2],0))
   self_edge[0] = np.dot( c[0,:] , self_center[0:2] )
   c = np.transpose(weights(ze[Nc-2],zc[Ne-3:Ne-1],0))
   self_edge[Nc-2] = np.dot( c[0,:] , self_center[Ne-3:Ne-1] )

 return self_edge

"""
def interp_to_edges( self_center , ze , zc ):
 Ne = np.shape(ze)[0] # edge number
 Nc = np.shape(zc)[0] # center number
 #self_edge = np.zeros([Nc-1])
 #for j in range(0,Nc-1):
 self_edge = np.interp(ze,zc,self_center)
 return self_edge 
"""

def interp_to_centers( self_edge , zc , ze ):
 Nc = np.shape(zc)[0]
 Ne = np.shape(ze)[0]
 self_center = np.zeros([Nc])
 c = np.transpose(weights(zc[0],ze[0:4],4))
 self_center[0] = np.dot( c[0,:] , self_edge[0:4] )
 c = np.transpose(weights(zc[Nc-1] , ze[Ne-5:Ne-1],4))
 self_center[Nc-1] = np.dot( c[0,:] , self_edge[0:4] )
 for j in range(1,Nc-1):
  c = np.transpose(weights(zc[j] , ze[j-1:j+3],0))
  #print(np.shape(c))
  #print(ze[j-1:j+2])
  self_center[j] = np.dot( c[0,:] , self_edge[j-1:j+3] )
  #self_center[j] = fnbg( zc[j] , ze , self , 4 , 0 )
 return self_center


def weights(z,x,m):
# From Bengt Fornbergs (1998) SIAM Review paper.
#  	Input Parameters
#	z location where approximations are to be accurate,
#	x(0:nd) grid point locations, found in x(0:n)
#	n one less than total number of grid points; n must
#	not exceed the parameter nd below,
#	nd dimension of x- and c-arrays in calling program
#	x(0:nd) and c(0:nd,0:m), respectively,
#	m highest derivative for which weights are sought,
#	Output Parameter
#	c(0:nd,0:m) weights at grid locations x(0:n) for derivatives
#	of order 0:m, found in c(0:n,0:m)
#      	dimension x(0:nd),c(0:nd,0:m)

  n = np.shape(x)[0]-1
  c = np.zeros([n+1,m+1])
  c1 = 1.0
  c4 = x[0]-z
  for k in range(0,m+1):  
    for j in range(0,n+1): 
      c[j,k] = 0.0
  c[0,0] = 1.0
  for i in range(0,n+1):
    mn = min(i,m)
    c2 = 1.0
    c5 = c4
    c4 = x[i]-z
    for j in range(0,i):
      c3 = x[i]-x[j]
      c2 = c2*c3
      if (j == i-1):
        for k in range(mn,0,-1): 
          c[i,k] = c1*(k*c[i-1,k-1]-c5*c[i-1,k])/c2
      c[i,0] = -c1*c5*c[i-1,0]/c2
      for k in range(mn,0,-1):
        c[j,k] = (c4*c[j,k]-k*c[j,k-1])/c3
      c[j,0] = c4*c[j,0]/c3
    c1 = c2
  return c



def grid_check_zoom(A,z,Ae,ze,Ac,zc,fig_title,fig_xlabel,fig_plotname,fig_axis1,fig_axis2,fig_axis3,log_flag,range_number):
 # plots topmost, middle, and bottommost points for interpolation inspection
 
 R = range_number # number of grid points to inspect
 Na = np.shape(z)[0]
 Nb = np.shape(ze)[0]
 Nc = np.shape(zc)[0]
 range1 = range(0,R)
 range1c = range(0,R-1)
 range2a = range(int(Na/2)-R-2,int(Na/2)-1)
 range2b = range(int(Nb/2)-R+2,int(Nb/2)+2)
 range2c = range(int(Nc/2)-R+2,int(Nc/2)+2)
 range3a = range(int(Na)-R-1,int(Na))
 range3b = range(int(Nb)-R-1,int(Nb))
 range3c = range(int(Nc)-R,int(Nc))

 fig = plt.figure(figsize=(14,5))
 
 plt.subplot(1,3,1)
 if log_flag == 'semilogx':
  plt.semilogx(A[range1],z[range1],'r',label='measurement')
  plt.semilogx(Ae[range1],ze[range1],'xb',label='uniform edges')
  plt.semilogx(Ac[range1c],zc[range1c],'ok',label='uniform centers')
  plt.semilogx(Ae[0],ze[0],'xb',markersize=2,label='top')
 else:
  plt.plot(A[range1],z[range1],'r',label='measurement')
  plt.plot(Ae[range1],ze[range1],'xb',label='uniform edges')
  plt.plot(Ac[range1c],zc[range1c],'ok',label='uniform centers')
  plt.plot(Ae[0],ze[0],'xb',markersize=2,label='top')
 plt.ylabel('z'); 
 plt.legend(loc=3); 
 plt.xlabel(fig_xlabel); 
 plt.title(fig_title);
 plt.axis(fig_axis1) 
 plt.grid()

 plt.subplot(1,3,2)
 if log_flag == 'semilogx':
  plt.semilogx(A[range2a],z[range2a],'r',label='measurement') 
  plt.semilogx(Ae[range2b],ze[range2b],'xb',label='uniform edges')
  plt.semilogx(Ac[range2c],zc[range2c],'ok',label='uniform centers')
 else:
  plt.plot(A[range2a],z[range2a],'r',label='measurement') 
  plt.plot(Ae[range2b],ze[range2b],'xb',label='uniform edges')
  plt.plot(Ac[range2c],zc[range2c],'ok',label='uniform centers')
 plt.legend(loc=3); 
 plt.xlabel(fig_xlabel); 
 plt.title(fig_title);
 plt.axis(fig_axis2) 
 plt.grid()

 plt.subplot(1,3,3)
 if log_flag == 'semilogx':
  plt.semilogx(A[range3a],z[range3a],'r',label='measurement') 
  plt.semilogx(Ae[range3b],ze[range3b],'xb',label='uniform edges')
  plt.semilogx(Ac[range3c],zc[range3c],'ok',label='uniform centers')
  plt.semilogx(Ae[Nb-1],ze[Nb-1],'xb',markersize=2,label='bottom')
 else:
  plt.plot(A[range3a],z[range3a],'r',label='measurement') 
  plt.plot(Ae[range3b],ze[range3b],'xb',label='uniform edges')
  plt.plot(Ac[range3c],zc[range3c],'ok',label='uniform centers')
  plt.plot(Ae[Nb-1],ze[Nb-1],'xb',markersize=2,label='bottom')
 plt.legend(loc=2); 
 plt.xlabel(fig_xlabel); 
 plt.title(fig_title);
 plt.axis(fig_axis3) 
 plt.grid()

 plt.savefig(fig_plotname,format="png"); 
 plt.close(fig)

 return

def grid_check(A,z,Ae,ze,Ac,zc,fig_title,fig_xlabel,fig_plotname,fig_axis,log_flag):
 # plots the entire profile for interpolation inspection
 
 fig = plt.figure(figsize=(7,5)) 

 if log_flag == 'semilogx':
  plt.semilogx(A,z,'r',label='measurement')
  plt.semilogx(Ae,ze,'--b',label='uniform edges')
  plt.semilogx(Ac,zc,'--r',label='uniform centers')
 else:
  plt.plot(A,z,'r',label='measurement')
  plt.plot(Ae,ze,'--b',label='uniform edges')
  plt.plot(Ac,zc,'--r',label='uniform centers')
 plt.ylabel('z'); 
 plt.legend(loc=3); 
 plt.xlabel(fig_xlabel); 
 plt.title(fig_title);
 plt.axis(fig_axis) 
 plt.grid()

 plt.savefig(fig_plotname,format="png"); 
 plt.close(fig)

 return

def remove_bad_eps( eps ):
 Neps = np.shape(eps)[0]
 #print(Neps)
 for k in range(0,Neps):
  if abs(eps[k]) >= 1e-2:
    eps[k] = np.nan
  if eps[k] < 0.:
    eps[k] = np.nan
 return eps

def remove_bad_SA( SA ):
 Nsa = np.shape(SA)[0]
 #print(N)
 for k in range(0,Nsa):
  if abs(SA[k]) >= 50.:
    SA[k] = np.nan
  if SA[k] < 0.:
    SA[k] = np.nan
 return SA

def remove_bad_CT( CT ):
 Nct = np.shape(CT)[0]
 #print(N)
 for k in range(0,Nct):
  if abs(CT[k]) >= 50.:
    CT[k] = np.nan
  if CT[k] <= -50.:
    CT[k] = np.nan
 return CT

def remove_bad_N2( N2 ):
 Nn2 = np.shape(N2)[0]
 #print(N)
 for k in range(0,Nn2):
  if abs(N2[k]) >= 1e-1:
    N2[k] = np.nan
  if N2[k] <= -1e-1:
    N2[k] = np.nan
 return N2


def get_hydro(my_file,count):
 f = Dataset(my_file, mode='r')
 
 eps = f.variables['EPSILON'][:]
 eps = remove_bad_eps( eps )
 lat = f.variables['LATITUDE'][:]
 lon = f.variables['LONGITUDE'][:]
 p = f.variables['PRESSURE'][:]
 SP = f.variables['PSAL'][:]
 T = f.variables['TEMPERATURE'][:]

 z = gsw.z_from_p(p,lat) # m
 SA = gsw.SA_from_SP(SP,p,lon,lat) #  g/kg, absolute salinity
 CT = gsw.CT_from_t(SA,T,p) # C, conservative temperature

 SA = remove_bad_SA( SA )
 CT = remove_bad_CT( CT )

 [N2_mid, p_mid] = gsw.Nsquared(SA,CT,p,lat)
 z_mid = gsw.z_from_p(p_mid,lat)

 N2 = interp_to_edges( N2_mid , z , z_mid , 4)
 #N2 = np.append(np.append([np.nan],N2),[np.nan])
 N2 = np.append(np.append(N2,[np.nan]),[np.nan])
 #eps_mid = interp_to_centers( eps , z_mid , z )
 #SA_mid = interp_to_centers( SA , z_mid , z )
 #CT_mid = interp_to_centers( CT , z_mid , z )

 N2 = remove_bad_N2( N2 )

 plotname = figure_path +'N2_%i.png' %(count)
 fig = plt.figure()
 plt.plot(N2_mid,z_mid,'r')#,label="computed")
 plt.plot(N2,z,'--b')#,label="computed")
 plt.axis('tight') #[-0.0001,0.0005,-35.,-23.])
 #plt.grid()
 plt.savefig(plotname,format="png"); plt.close(fig);

 plotname = figure_path +'SA_%i.png' %(count)
 fig = plt.figure()
 #plt.plot(SA_mid,z_mid,'b')#,label="computed")
 plt.plot(SA,z,'b')#,label="computed")
 plt.axis('tight') #[-0.0001,0.0005,-35.,-23.])
 #plt.grid()
 plt.savefig(plotname,format="png"); plt.close(fig);

 plotname = figure_path +'CT_%i.png' %(count)
 fig = plt.figure()
 #plt.plot(CT_mid,z_mid,'b')#,label="computed")
 plt.plot(CT,z,'b')#,label="computed")
 plt.axis('tight') #[-0.0001,0.0005,-35.,-23.])
 #plt.grid()
 plt.savefig(plotname,format="png"); plt.close(fig);

 plotname = figure_path +'eps_%i.png' %(count)
 fig = plt.figure()
 #plt.plot(eps_mid,z_mid,'b')#,label="computed")
 plt.semilogx(eps,z,'b')#,label="computed")
 plt.axis('tight') #[-0.0001,0.0005,-35.,-23.])
 #plt.grid()
 plt.savefig(plotname,format="png"); plt.close(fig);

 return N2, SA, CT, eps, z


def throw_points( A, N2, SA, CT, eps, z ):
 locs = np.argwhere(np.isnan(A)-1)[:,0]
 #print(locs)
 #print(N2[locs])
 N2 = N2[locs]
 SA = SA[locs]
 CT = CT[locs]
 eps = eps[locs]
 z = z[locs]
 return N2, SA, CT, eps, z

def nanrid( N2, SA, CT, eps, z ):
 # get rid of nans!
 [N2, SA, CT, eps, z] = throw_points( N2, N2, SA, CT, eps, z )
 [N2, SA, CT, eps, z] = throw_points( SA, N2, SA, CT, eps, z )
 [N2, SA, CT, eps, z] = throw_points( CT, N2, SA, CT, eps, z )
 [N2, SA, CT, eps, z] = throw_points( eps, N2, SA, CT, eps, z )
 return N2, SA, CT, eps, z

def contour_plots( N2, SA, CT, eps ):

 # scatter plots

 Ngrid = 4000
 [SAg,CTg] = np.meshgrid(np.linspace(np.amin(SA),np.amax(SA),num=Ngrid),np.linspace(np.amin(CT),np.amax(CT),num=Ngrid))
 sigma4 = gsw.sigma4(SAg,CTg)
 sigma3 = gsw.sigma3(SAg,CTg)
 sigma2 = gsw.sigma3(SAg,CTg)
 sigma1 = gsw.sigma1(SAg,CTg)
 sigma0 = gsw.sigma0(SAg,CTg)

 plotname = figure_path +'contour_SA_CT_sig4.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.contourf(SAg,CTg,sigma4,50)
 plt.scatter(SA,CT,marker='o', color='black')
 plt.xlabel(r"S$_A$",fontsize=13);
 plt.ylabel(r"$\Theta$",fontsize=13); 
 plt.title(r"$\sigma_4$ contours")
 #plt.grid(True)
 plt.axis([np.amin(SA),np.amax(SA),np.amin(CT),np.amax(CT)])
 plt.savefig(plotname,format="png"); plt.close(fig);

 plotname = figure_path +'contour_SA_CT_sig3.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.contourf(SAg,CTg,sigma3,50)
 plt.scatter(SA,CT,marker='o', color='black')
 plt.xlabel(r"S$_A$",fontsize=13);
 plt.ylabel(r"$\Theta$",fontsize=13); 
 plt.title(r"$\sigma_3$ contours")
 #plt.grid(True)
 plt.axis([np.amin(SA),np.amax(SA),np.amin(CT),np.amax(CT)])
 plt.savefig(plotname,format="png"); plt.close(fig);

 plotname = figure_path +'contour_SA_CT_sig2.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.contourf(SAg,CTg,sigma2,50)
 plt.scatter(SA,CT,marker='o', color='black')
 plt.xlabel(r"S$_A$",fontsize=13);
 plt.ylabel(r"$\Theta$",fontsize=13); 
 plt.title(r"$\sigma_2$ contours")
 #plt.grid(True)
 plt.axis([np.amin(SA),np.amax(SA),np.amin(CT),np.amax(CT)])
 plt.savefig(plotname,format="png"); plt.close(fig); 

 plotname = figure_path +'contour_SA_CT_sig1.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.contourf(SAg,CTg,sigma1,50)
 plt.scatter(SA,CT,marker='o', color='black')
 plt.xlabel(r"S$_A$",fontsize=13);
 plt.ylabel(r"$\Theta$",fontsize=13); 
 plt.title(r"$\sigma_1$ contours")
 #plt.grid(True)
 plt.axis([np.amin(SA),np.amax(SA),np.amin(CT),np.amax(CT)])
 plt.savefig(plotname,format="png"); plt.close(fig);

 plotname = figure_path +'contour_SA_CT_sig0.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.contourf(SAg,CTg,sigma0,50)
 plt.scatter(SA,CT,marker='o', color='black')
 plt.xlabel(r"S$_A$",fontsize=13);
 plt.ylabel(r"$\Theta$",fontsize=13); 
 plt.title(r"$\sigma_0$ contours")
 #plt.grid(True)
 plt.axis([np.amin(SA),np.amax(SA),np.amin(CT),np.amax(CT)])
 plt.savefig(plotname,format="png"); plt.close(fig);

 plotname = figure_path +'scatter_eps_N2.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.scatter(eps/np.nanmean(eps),N2/np.nanmean(N2),marker='o', color='blue')#,label="computed")
 plt.xlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13);
 plt.ylabel(r"$N^2/\overline{N^2}$",fontsize=13); 
 plt.grid(True)
 plt.axis([-50.,4000.,-300.,250.])
 plt.savefig(plotname,format="png"); plt.close(fig); 

 plotname = figure_path +'scatter_CT_SA.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.scatter(CT,SA,marker='o', color='blue')#,label="computed")
 plt.xlabel(r"$\Theta$",fontsize=13);
 plt.ylabel(r"S$_A$",fontsize=13); 
 plt.grid(True)
 plt.axis('tight') #[-50.,4000.,-0.25,3.])
 plt.savefig(plotname,format="png"); plt.close(fig);

 plotname = figure_path +'scatter_eps_CT.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.scatter(eps/np.nanmean(eps),CT/np.nanmean(CT),marker='o', color='blue')#,label="computed")
 plt.xlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13);
 plt.ylabel(r"$\Theta/\overline{\Theta}$",fontsize=13); 
 plt.grid(True)
 plt.axis([-50.,4000.,-0.25,3.])
 plt.savefig(plotname,format="png"); plt.close(fig); 

 plotname = figure_path +'scatter_eps_SA.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.scatter(eps/np.nanmean(eps),SA/np.nanmean(SA),marker='o', color='blue')#,label="computed")
 plt.xlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13);
 plt.ylabel(r"$S_A/\overline{S}_A$",fontsize=13); 
 plt.grid(True)
 plt.axis([-50.,4000.,0.6,1.1])
 plt.savefig(plotname,format="png"); plt.close(fig); 

 nu = 1e-6
 Reb = eps/(nu*N2)
 plotname = figure_path +'scatter_eps_Reb.png' #%(start_time,end_time)
 fig = plt.figure()
 plt.scatter(eps/np.nanmean(eps),Reb,marker='o', color='blue')#,label="computed")
 plt.xlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13);
 plt.ylabel(r"Re$_b$",fontsize=13); 
 plt.axis([-50.,4000.,-1e9,1e9])
 plt.grid(True)
 plt.savefig(plotname,format="png"); plt.close(fig); 

 return

def pdf( self ):
 
 Ns = np.shape(self)[0]
 mu = np.mean(self)
 
 sig = np.std(self)
 #sig2 = np.sqrt(np.sum((self - mu*np.ones([Ns]))**2.)/Ns) # def of np.std
 sk = (np.sqrt(np.sum((self - mu*np.ones([Ns]))**3.)/Ns))/(sig**3.)
 fl = (np.sqrt(np.sum((self - mu*np.ones([Ns]))**4.)/Ns))/(sig**4.) - 3.

 smin = np.amin(self)
 smax = np.amax(self)
 print(mu,sig,sk,fl,smin,smax)
 """
 print(np.mean(eps),np.std(eps))
 #print(np.mean(CT),np.std(CT))
 #print(np.mean(SA),np.std(SA))
 #print(np.mean(N2),np.std(N2))

 N = length(u); 
 nsigma = zeros(1,N); nskew = nsigma; nflat = nsigma;
 mu = mean(u); 
 for i = 1:N
       nsigma(i) = (u(i)-mu)^2;
       nskew(i) = (u(i)-mu)^3;
       nflat(i) = (u(i)-mu)^4;

 var = (sum(nsigma))/N;  % variance
 sigma = sqrt(var); % standard deviation
 skew = (sum(nskew)/N)/(sigma^3); % skewness
 flat = (sum(nflat)/N)/(sigma^4)-3; % flatness
 minimum = min(u); % min
 maximum = max(u); % max
 """
 return mu,sig,sk,fl,smin,smax


def raw_pdf_plot( N2, SA, CT, eps ):
 
 binsize = int((np.log10(np.amax(eps))-np.log10(np.amin(eps)))/0.05)
 plotname = figure_path +'histogram_eps_raw.png' 
 fig = plt.figure(figsize=(8,5))
 plt.hist(np.log10(eps), color = 'blue', edgecolor = 'black',
         bins = binsize)
 plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
 plt.ylabel(r"number of measurements",fontsize=13)
 #ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
 plt.savefig(plotname,format="png"); plt.close(fig);

 binsize = int((np.amax(CT)-np.amin(CT))/0.2)
 plotname = figure_path +'histogram_CT_raw.png' 
 fig = plt.figure(figsize=(8,5))
 plt.hist(CT, color = 'blue', edgecolor = 'black',
         bins = binsize)
 plt.xlabel(r"$\Theta$",fontsize=13)
 plt.ylabel(r"number of measurements",fontsize=13)
 #ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
 plt.savefig(plotname,format="png"); plt.close(fig);

 binsize = int((np.amax(SA)-np.amin(SA))/0.1)
 plotname = figure_path +'histogram_SA_raw.png' 
 fig = plt.figure(figsize=(8,5))
 plt.hist(SA, color = 'blue', edgecolor = 'black',
         bins = binsize)
 plt.xlabel(r"S$_A$",fontsize=13)
 plt.ylabel(r"number of measurements",fontsize=13)
 #ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
 plt.savefig(plotname,format="png"); plt.close(fig) 

 binsize = int( abs(np.amax(N2)-np.amin(N2)) / 0.0002)
 plotname = figure_path +'histogram_N2_raw.png' 
 fig = plt.figure(figsize=(8,5))
 plt.hist(np.log10(N2), color = 'blue', edgecolor = 'black',
         bins = binsize)
 plt.xlabel(r"log$_{10}(N^2)$",fontsize=13)
 plt.ylabel(r"number of measurements",fontsize=13)
 #ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
 plt.savefig(plotname,format="png"); plt.close(fig);

 return


def remove_outliers( N2, SA, CT, eps ):
 Np = np.shape(eps)[0]

 for j in range(0,Np):
  if eps[j] >= 1e-4:
   eps[j] = np.nan
  if eps[j] <= 1e-12:
   eps[j] = np.nan
 [N2, SA, CT, eps] = nanrid( N2, SA, CT, eps )
 Np = np.shape(eps)[0]

 for j in range(0,Np):
  if abs(N2[j]) >= 1e-2:
   N2[j] = np.nan
  if abs(N2[j]) <= 1e-9:
   eps[j] = np.nan
 [N2, SA, CT, eps] = nanrid( N2, SA, CT, eps )
 Np = np.shape(eps)[0]

 for j in range(0,Np):
  if SA[j] >= 38:
   SA[j] = np.nan
  if SA[j] <= 33.:
   SA[j] = np.nan
 [N2, SA, CT, eps] = nanrid( N2, SA, CT, eps )
 Np = np.shape(eps)[0]

 for j in range(0,Np):
  if CT[j] >= 30:
   CT[j] = np.nan
  if CT[j] <= 0.:
   CT[j] = np.nan
 [N2, SA, CT, eps] = nanrid( N2, SA, CT, eps )
 Np = np.shape(eps)[0]

 return N2, SA, CT, eps

def pdf_plot( N2, SA, CT, eps ):
 
 binsize = int((np.log10(np.amax(eps))-np.log10(np.amin(eps)))/0.05)
 plotname = figure_path +'histogram_eps.png' 
 fig = plt.figure(figsize=(8,5))
 plt.hist(np.log10(eps), color = 'blue', edgecolor = 'black',
         bins = binsize)
 plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
 plt.ylabel(r"number of measurements",fontsize=13)
 #ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
 plt.savefig(plotname,format="png"); plt.close(fig);

 binsize = int((np.amax(CT)-np.amin(CT))/0.2)
 plotname = figure_path +'histogram_CT.png' 
 fig = plt.figure(figsize=(8,5))
 plt.hist(CT, color = 'blue', edgecolor = 'black',
         bins = binsize)
 plt.xlabel(r"$\Theta$",fontsize=13)
 plt.ylabel(r"number of measurements",fontsize=13)
 #ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
 plt.savefig(plotname,format="png"); plt.close(fig);

 binsize = int((np.amax(SA)-np.amin(SA))/0.1)
 plotname = figure_path +'histogram_SA.png' 
 fig = plt.figure(figsize=(8,5))
 plt.hist(SA, color = 'blue', edgecolor = 'black',
         bins = binsize)
 plt.xlabel(r"S$_A$",fontsize=13)
 plt.ylabel(r"number of measurements",fontsize=13)
 #ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
 plt.savefig(plotname,format="png"); plt.close(fig) 

 binsize = int( abs(np.amax(N2)-np.amin(N2)) / 0.0002)
 plotname = figure_path +'histogram_N2.png' 
 fig = plt.figure(figsize=(8,5))
 plt.hist(np.log10(N2), color = 'blue', edgecolor = 'black',
         bins = binsize)
 plt.xlabel(r"log$_{10}(N^2)$",fontsize=13)
 plt.ylabel(r"number of measurements",fontsize=13)
 #ax.set_zlabel(r"$\varepsilon/\overline{\varepsilon}$",fontsize=13)
 plt.savefig(plotname,format="png"); plt.close(fig);

 return
