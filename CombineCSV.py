#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

def main():
	
	for f in os.listdir(os.getcwd()):
		if f.find('result')!=-1:
			data = np.loadtxt(f,dtype=float,delimiter=',',unpack=True)
			
			try:
				newdata.append(data[-1])
			except:
				newdata = []
				newdata.append(data[0])		
				newdata.append(data[-1])		
			
	newdata = np.array(newdata).transpose()
		
	A = newdata[:,1:]
	Av = np.average(A,axis=1)
	std = np.std(A,axis=1)
	
	output = np.array([newdata[:,0],Av,std]).transpose()
	np.savetxt('data.csv',output,delimiter=',')
			
	return 0

if __name__ == '__main__':
	main()

