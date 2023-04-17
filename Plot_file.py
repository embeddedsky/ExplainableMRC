import raw_read
import subprocess as sp
from matplotlib import pyplot
import numpy as np
from files import *
from sklearn.metrics import mean_squared_error

[arrs1,plots]=raw_read.rawread('output/rawfile.raw')
# time
stime = arrs1[0]
res_out = arrs1[1:-1]
x=res_out.T
temp = np.linalg.pinv(x)
vref=np.load("output/ref.npy")
vref=vref.T
w_out = np.load("output/w_out.npy")
outlayer=1

readout=np.zeros((res_out.shape[1],outlayer))
for t in range(0, res_out.shape[1]):
     # readout[t]=np.dot(res[:,:,t],w_out)
    readout[t] = np.dot(res_out[:, t], w_out)

# error = abs(readout - vref)
error = abs(readout[1:] - vref[1:])
error1= mean_squared_error(readout[1:], vref[1:],squared=False)


print(error1)

pyplot.figure(figsize=(10,6))
pyplot.subplots_adjust(wspace =0, hspace =0.2)
pyplot.plot(stime[1:],vref[1:],'r')
pyplot.plot(stime[1:],readout[1:],'mediumspringgreen')

pyplot.xlabel("Time (s)",fontsize=15)
pyplot.show()






###################################plot########################################
# stime = arrs1[0]
# current = arrs1[1:19]
# memstate=arrs1[21:39]
#
# current1=arrs1[1]
# state1=arrs1[21]
#
#
# pyplot.figure(figsize=(10,3))
# pyplot.subplots_adjust(wspace =0, hspace =0.2)
# # pyplot.plot(stime[1:],current1[1:],'r')
# pyplot.plot(stime[1:],state1[1:],'mediumspringgreen')
#
# pyplot.xlabel("Time (s)",fontsize=15)
# pyplot.show()
#
# pyplot.figure(figsize=(10,3))
# pyplot.subplots_adjust(wspace =0, hspace =0.2)
# pyplot.plot(stime[1:],current1[1:],'r')
# # pyplot.plot(stime[1:],state1[1:],'mediumspringgreen')
# pyplot.xlabel("Time (s)",fontsize=15)
# pyplot.show()