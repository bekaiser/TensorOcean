# Mixture density networks for epsilon prediction from individual .nc files
# Bryan Kaiser
# 2/16/19

import matplotlib
matplotlib.use('Agg') # set non-interactive backend for PNGs, called before .pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D # note the capitalization
import tensorflow as tf
import numpy as np
import math as ma
import functions as fn
from netCDF4 import Dataset
import gsw
import os
from os import listdir
from os.path import isfile, join
#import h5py
#import array

figure_path = "./figures/"
data_path = "/home/bryan/data/TensorOcean/UCSD_microstructure/bbtre97_microstructure"
alpha_level = 0.1


# =============================================================================
# get training data 

N2 = np.empty([0])
SA = np.empty([0])
CT = np.empty([0])
eps = np.empty([0])
z = np.empty([0])


# loop over folder path
count = 0
plot_filenames = []


onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
Nfiles = np.shape(onlyfiles)[0] # number of files (time steps)

Nfiles = 60

for j in range(0,Nfiles):
   my_file = data_path + '/' + onlyfiles[j] 
   print('file =', my_file)
   [N2j, SAj, CTj, epsj, zj] = fn.get_hydro(my_file,count)
   [N2j, SAj, CTj, epsj, zj] = fn.nanrid( N2j, SAj, CTj, epsj, zj )
   [N2j, SAj, CTj, epsj, zj] = fn.remove_outliers( N2j, SAj, CTj, epsj, zj )
   # do bin means
   N2=np.concatenate((N2,N2j),axis=0)
   SA=np.concatenate((SA,SAj),axis=0)
   CT=np.concatenate((CT,CTj),axis=0)
   eps=np.concatenate((eps,epsj),axis=0)
   z=np.concatenate((z,zj),axis=0)
   count = count + 1
   #plot_filenames = np.append(plot_filenames,my_file)

# add distribution

NMULTI = 4
NSAMPLE = np.shape(N2)[0]

plotname = figure_path +'training_data.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(np.log10(eps),z,'ko',alpha=alpha_level)
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"$N_{samples}=$%i" %(NSAMPLE),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

X_DATA = np.zeros([NSAMPLE,NMULTI])
X_DATA[:,0] = N2[:]
X_DATA[:,1] = z[:]
X_DATA[:,2] = CT[:]
X_DATA[:,3] = SA[:]
#print(np.shape(X_DATA))

y_data = np.zeros([NSAMPLE,1])
y_data[:,0] = np.log10(eps[:])
print(np.shape(y_data))

z_train = z

# =============================================================================
# get test data 

N2 = np.empty([0])
SA = np.empty([0])
CT = np.empty([0])
eps = np.empty([0])
z = np.empty([0])


# loop over folder path
count = 0
plot_filenames = []


onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
Nfiles = np.shape(onlyfiles)[0] # number of files (time steps)

Nfiles = 1

for j in range(42,Nfiles+42):
   my_file = data_path + '/' + onlyfiles[j] 
   print('file =', my_file)
   [N2j, SAj, CTj, epsj, zj] = fn.get_hydro(my_file,count)
   [N2j, SAj, CTj, epsj, zj] = fn.nanrid( N2j, SAj, CTj, epsj, zj )
   [N2j, SAj, CTj, epsj, zj] = fn.remove_outliers( N2j, SAj, CTj, epsj, zj )
   # do bin means
   N2=np.concatenate((N2,N2j),axis=0)
   SA=np.concatenate((SA,SAj),axis=0)
   CT=np.concatenate((CT,CTj),axis=0)
   eps=np.concatenate((eps,epsj),axis=0)
   z=np.concatenate((z,zj),axis=0)
   count = count + 1
   #plot_filenames = np.append(plot_filenames,my_file)

# add distribution

NMULTI = 4
NSAMPLE = np.shape(N2)[0]

plotname = figure_path +'test_data.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(np.log10(eps),z,'ko',alpha=alpha_level)
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"$N_{samples}=$%i" %(NSAMPLE),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

X_TEST = np.zeros([NSAMPLE,NMULTI])
X_TEST[:,0] = N2[:]
X_TEST[:,1] = z[:]
X_TEST[:,2] = CT[:]
X_TEST[:,3] = SA[:]
#print(np.shape(X_DATA))

y_test = np.zeros([NSAMPLE,1])
y_test[:,0] = np.log10(eps[:])
print(np.shape(y_test))

z_test = z


# =============================================================================
# A mixture density network


# construct the MDN:
NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,NMULTI], name="x") 
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([NMULTI,NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer,Wo) + bo

out_pi, out_sigma, out_mu = fn.get_mixture_coeff(output,KMIX)

lossfunc = fn.get_lossfunc(out_pi, out_sigma, out_mu, y)
train_op = tf.train.AdamOptimizer().minimize(lossfunc)

# training:
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# = 6000 for NHIDDEN = 24, STDEV = 0.5, KMIX = 24
NEPOCH = 2000 
loss = np.zeros(NEPOCH) # store the training progress here.
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: X_DATA, y: y_data})
  loss[i] = sess.run(lossfunc, feed_dict={x: X_DATA, y: y_data})

plotname = figure_path +'convergence_3d_mdn.png' 
plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'b-')
plt.ylabel(r"loss",fontsize=13)
plt.xlabel(r"$N_{epoch}$",fontsize=13)
plt.title(r"$N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i, $N_{mix}=$%i" %(NSAMPLE,NHIDDEN,NEPOCH,KMIX),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


# training data prediction:
out_pi_train, out_sigma_train, out_mu_train = sess.run(fn.get_mixture_coeff(output,KMIX), feed_dict={x: X_DATA})
y_train_pred = fn.generate_ensemble( out_pi_train, out_mu_train, out_sigma_train, X_DATA , 1 )

#print(np.shape(out_pi_train), np.shape(out_mu_train), np.shape(out_sigma_train))

# test data prediction:
out_pi_test, out_sigma_test, out_mu_test = sess.run(fn.get_mixture_coeff(output,KMIX), feed_dict={x: X_TEST})
y_test_pred = fn.generate_ensemble( out_pi_test, out_mu_test, out_sigma_test, X_TEST , 1 )

#print(np.shape(out_pi_test), np.shape(out_mu_test), np.shape(out_sigma_test))

sess.close()

plotname = figure_path +'training_prediction_mdn.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(y_data,z_train,'ko',alpha=alpha_level,label=r"training data")
plt.plot(y_train_pred,z_train,'bo',alpha=alpha_level,label=r"predictions")
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"$N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i, $N_{mix}=$%i" %(NSAMPLE,NHIDDEN,NEPOCH,KMIX),fontsize=13)
plt.axis([-12.,-6.,-5500.,200.])
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'test_prediction_mdn.png' 
fig = plt.figure(figsize=(8, 12))
plt.plot(y_test,z_test,'ko',alpha=alpha_level,label=r"test data")
plt.plot(y_test_pred,z_test,'bo',alpha=alpha_level,label=r"predictions")
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"$N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i, $N_{mix}=$%i" %(NSAMPLE,NHIDDEN,NEPOCH,KMIX),fontsize=13)
plt.axis([-12.,-6.,-5200.,0.])
plt.savefig(plotname,format="png"); plt.close(fig);


Nwindow = 40
z_test_smooth = fn.smooth(z_test,Nwindow,'hanning')
y_test_smooth = fn.smooth(y_test[:,0],Nwindow,'hanning')
y_test_pred_smooth = fn.smooth(y_test_pred[:,0],Nwindow,'hanning')

plotname = figure_path +'test_prediction_mdn_smooth.png' 
fig = plt.figure(figsize=(8, 12))
plt.plot(y_test_smooth,z_test_smooth,'k',alpha=0.9,label=r"test data")
plt.plot(y_test_pred_smooth,z_test_smooth,'b',alpha=0.9,label=r"predictions")
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"$N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i, $N_{mix}=$%i" %(NSAMPLE,NHIDDEN,NEPOCH,KMIX),fontsize=13)
plt.axis([-12.,-6.,-5200.,0.])
plt.savefig(plotname,format="png"); plt.close(fig);
