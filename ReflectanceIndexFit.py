#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib as mpl
from matplotlib import pyplot as plt

import putil
import sys
import os

import TMM as tm

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'


def main():

	'''
	thicknesses in microns (um)
	'''

	eps = putil.getEps('al',1500.)
	print eps
	exit()

	# Set layer indices	
	ni = 1.
	nf = 4.
	nt = sqrt(-100+1j)

	ns_film = np.array([nf])
	ns_bare = np.array([ni])
	
	# Set layer thicknesses
	ds = np.array([1]) # um
	
	# Set excitation propeties
	wls = np.linspace(1.,2.,100) # um
	# ths = np.linspace(0.0, 90, 1000) * pi/180.
	ths = np.array([45.*pi/180])
	pol = 'p'
	
	# Collect data
	R_bare = pl.zeros((len(wls),len(ths)), dtype='float')
	R_film = pl.zeros((len(wls),len(ths)), dtype='float')
	
	for ith, th in enumerate(ths):
		for iwl, wl in enumerate(wls):
			R_film[iwl,ith] = tm.solvestack(ni, nt, ns_film, ds, wl, pol, th)[0]
			R_bare[iwl,ith] = tm.solvestack(ni, nt, ns_bare, ds, wl, pol, th)[0]

	R = R_film/R_bare
	
	# Plot data
	pl.figure(figsize=(10,7.5))
	pl.plot(wls, R[:,0], 'r', lw=2, label='R')
	#pl.xlim(ths[0], ths[-1])
	#pl.ylim(0, 1)
	pl.ylabel(r'$R$', fontsize=18)
	pl.xlabel(r'$\theta$', fontsize=18)
	pl.show()
	

	return

if __name__ == '__main__':
  main()
