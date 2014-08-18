#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
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

	nm = 1e-9
	um = 1e-6
	cm = 1e-3

	wls = np.linspace(1500,6000,100)

	# e_agst = putil.getEps('a-gete', wls)
	# e_cgst = putil.getEps('c-gete', wls)
	e_agst = putil.getEps('a-gst225', wls)
	e_cgst = putil.getEps('c-gst225', wls)

	n_agst = sqrt(e_agst)
	n_agst = n_agst.real + 1j * n_agst.imag
	
	n_cgst = sqrt(e_cgst)
	n_cgst = n_cgst.real + 1j * n_cgst.imag
	
	e_al = putil.getEps('al',wls)
	n_al = sqrt(e_al)
	n_al = n_al.real + 1j * n_al.imag

	Nths = 10
	ths = np.linspace(10,40,Nths) * pi/180.
	pol = 'p'
	
	ds = np.array([175]) # nm
	
	R = np.zeros((len(wls),Nths,2), dtype='float')
	T = np.zeros((len(wls),Nths,2), dtype='float')
	A = np.zeros((len(wls),Nths,2), dtype='float')

	plt.figure(figsize=(5,4))

	for phase in ['a','c']:
		for ip, p in enumerate(['p','s']):
			for ith, th in enumerate(ths):
				for iwl, wl in enumerate(wls):

					ni = 1+0j
					ns_a = np.array([n_agst[iwl]])
					ns_c = np.array([n_cgst[iwl]])
					nt = n_al[iwl]

					# reflection from aluminum with no GST
					ref = tm.solvestack(ni, nt, np.array([1]), ds, wl, p, th)[0]

					if phase == 'a':
						R[iwl,ith,ip], T[iwl,ith,ip], A[iwl,ith,ip] = tm.solvestack(ni, nt, ns_a, ds, wl, p, th)	

					else:
						R[iwl,ith,ip], T[iwl,ith,ip], A[iwl,ith,ip] = tm.solvestack(ni, nt, ns_c, ds, wl, p, th)

					R[iwl,ith,ip] /= ref


		Rav = np.sum(np.sum(R,axis=1), axis=1) / (2*Nths)
		c = 'b' if phase == 'a' else 'r'
		plt.plot(wls/1e3, Rav, c, lw=2, label='R')	
		
	plt.ylabel(r'$R$', fontsize=18)
	plt.xlabel(r'$\lambda$ ($\mu m$)', fontsize=18)
	plt.legend(('amor.','crys.'), loc='best')
	plt.tight_layout()
	plt.show()
	
if __name__ == '__main__':
  main()

