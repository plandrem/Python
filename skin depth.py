#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

def main():
	
	e0 = 8.85e-12
	mu = 4*sp.pi*1e-7
	pi = sp.pi
	c = 3e8
	
	wp = 2e15 * 2*pi
	g = 4.4e12
	s = wp**2*e0/g
	print '%.3e' % s
	
	w = np.linspace(1e13,1e15,1000)
	k = sp.emath.sqrt((w/c)**2 + 1j*(s*w*mu/(1-1j*w/g)))
	d = sp.sqrt(2/(w*mu*s))
	
	plt.plot(w,1/k.imag,'r-')
	plt.plot(w,d,'b')
	plt.show()
	return 0

if __name__ == '__main__':
	main()

