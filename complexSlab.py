#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

import putil
import sys
import os

from EndFacet import beta_marcuse as Beta

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'


def main():

	'''
	Complex slab mode solver based on "Exact Solution To Guided Wave Propagation in Lossy Films" by Nagel et al,
	Optics Express 2011

	Currently supports TE even modes only.
	'''

	hlam = 0.5

	nf = 2 + 0.5j
	nc = 1.5

	wl = 1
	h = hlam*wl
	k = 2*pi/wl
	kf = nf*k
	kc = nc*k

	def f(kx):
		return kx*tan(kx * h) - sqrt(kf**2 - kc**2 - kx**2)

	def phi(wavevectors):
		a,b = wavevectors
		kx = b + 1j*a
		return np.real(f(kx)*f(kx).conj())

	def phiPlotter(kx):
		return np.real(10*sp.log10(f(kx)*f(kx).conj()))

	phiPlotter = np.vectorize(phiPlotter)

	# Find minimum residual near initial guess. Returns OptimizeResult object
	res = sp.optimize.minimize(phi,[0,7.5])
	
	if res.success:
		ax_min, bx_min = res.x
	else:
		print 'Minimize fail.'
		exit()

	kx_min = bx_min + 1j*ax_min

	# Find corresponding Beta (from equation A4)
	B = sqrt(kf**2-kx_min**2)
	print B

	# Pretty picture

	a = np.linspace(-1.5,1.5,100)
	b = np.linspace(0,10,100)

	A,B = np.meshgrid(a,b)

	kx = B+1j*A

	ext = np.array(putil.getExtent(b,a))
	asp = 'auto'

	plt.figure()
	plt.imshow(phiPlotter(kx).transpose(),extent=ext, vmax=40, vmin=-20, aspect=asp)
	plt.colorbar()
	plt.xlabel(r'$\beta_{x} \lambda_{o}$')
	plt.ylabel(r'$\alpha_{x}\lambda_{o}$')

	# Beta(n,d,wl=wl,pol=pol,polarity='even',Nmodes=None,plot=True)
	# fig.colorbar(ima)
	plt.show()



	return

if __name__ == '__main__':
  main()
