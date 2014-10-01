#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

def CompareGST():
	ev = np.linspace(0.1,2,1000)
	wl = putil.ev2nm(ev)

	c225 = putil.getEps('c-gst225',wl)
	a225 = putil.getEps('a-gst225',wl)

	c326 = putil.getEps('c-gst326',wl)
	a326 = putil.getEps('a-gst326',wl)

	cgete = putil.getEps('c-gete',wl)
	agete = putil.getEps('a-gete',wl)

	n_a225 = sp.emath.sqrt(a225)
	n_c225 = sp.emath.sqrt(c225)
	n_a326 = sp.emath.sqrt(a326)
	n_c326 = sp.emath.sqrt(c326)
	n_agete = sp.emath.sqrt(agete)
	n_cgete = sp.emath.sqrt(cgete)
	

	fig, ax = plt.subplots(2,figsize=(7,6), sharex=True)

	wl *= 1e-3

	ax[0].plot(wl,n_a225.real,'b:')
	ax[0].plot(wl,n_a326.real,'b')
	ax[0].plot(wl,n_agete.real,'b--')

	ax[1].plot(wl,n_a225.imag,'b:')
	ax[1].plot(wl,n_a326.imag,'b')
	ax[1].plot(wl,n_agete.imag,'b--')

	ax[0].plot(wl,n_c225.real,'r:')
	ax[0].plot(wl,n_c326.real,'r')
	ax[0].plot(wl,n_cgete.real,'r--')

	ax[1].plot(wl,n_c225.imag,'r:')
	ax[1].plot(wl,n_c326.imag,'r')
	ax[1].plot(wl,n_cgete.imag,'r--')

	plt.xlim(0.6, 10)

	ax[1].set_ylim(0,2)

	ax[0].set_ylabel('n')
	ax[1].set_ylabel('k')

	ax[1].set_xlabel(r'Wavelength ($\mu m$)')

	ax[1].legend(('a-GST225','a-GST326','a-GeTe','c-GST225','c-GST326','c-GeTe'))

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
  CompareGST()
