#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

def main():
	
	N = 64   #number of WDM channels
	B = 1e12 #bit rate per channel
	
	beta = lambda L: N*B / (0.12*sp.log(56.6*L/pi))
	
	L = np.linspace(0,5e3,1000)
	
	bs = np.array(map(beta,L))
	
	plt.plot(L*1e-3,bs*1e-12,color='r',lw=2)
	
	fs = 17
	plt.xlabel('Interconnect Length (mm)',fontsize=fs)
	plt.ylabel(r'Linear Bandwidth Density (Tb/s/$\mu$m)',fontsize=fs) 
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

