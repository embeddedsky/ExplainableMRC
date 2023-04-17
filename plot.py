import CRJ_Genome
import numpy as np
import sys
import pickle
import numpy as np
from files import *
import subprocess as sp
from raw_read_fuwuqi import  rawread
import random
import re
import sys
from matplotlib import  pyplot

def test_cal_fitness(g,folder):
    vref = np.load(folder + "/ref.npy")
    vref = vref.T

    g.change_to_cir(folder)
    p = sp.Popen(["%s" % (spicepath), '-b', '-o', log_file, folder + cir_filename])
    try:
        p.communicate(timeout=1000)
    except:
        p.kill()
        print("program run error")
    p.kill()

    [arrs, plots] = rawread(target_filename)  # arrs is the voltages'n'currents
    res_out = arrs[0:-1]
    x=res_out.T
    temp = np.linalg.pinv(x)
    w_out = np.dot(temp, vref)
    readout=np.zeros((res_out.shape[1],1))
    for t in range(0, res_out.shape[1]):
        readout[t] = np.dot(res_out[:, t], w_out)
    fitness=-9999
    error=abs(readout[:,0]-vref)
    error1=np.sqrt(np.sum(np.power(error,2))/res_out.shape[1])

    pyplot.figure(figsize=(10,4.5))
    pyplot.grid()
    pyplot.plot(readout[:,0],'r',marker='o', label='act. curve')
    pyplot.plot(vref,'g', marker='*',label='ref. curve')
    pyplot.savefig('one.png')

    fitness=100-(10000*error1)

    NMSE=np.sum(np.power(readout[:,0]-vref,2))/np.sum(np.power(vref,2))
    with open("outputnmse", 'w+') as file:
        file.writelines(str(NMSE))
        file.writelines('\n')
        file.writelines(str(fitness))

with open(sys.argv[3] + "/" + "best_genome.pkl", 'rb') as file:
    genome = pickle.loads(file.read())
    test_cal_fitness(genome,"Narma/Train")


