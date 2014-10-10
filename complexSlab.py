#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

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

	f_vector = np.vectorize(f)

	def phi(wavevectors):
		b,a = wavevectors
		kx = b + 1j*a
		return np.real(f(kx)*f(kx).conj())

	def phiPlotter(kx):
		return np.real(10*sp.log10(f(kx)*f(kx).conj()))

	phiPlotter = np.vectorize(phiPlotter)

	'''
	Find zeros in residue, convert to propagation constants
	'''

	# Generate residue for complex plane

	a = np.linspace(-1.5,1.5,20)
	b = np.linspace(0,10,20)

	B,A = np.meshgrid(b,a)

	kx = B+1j*A


	# Look for sign changes in real and imaginary parts of f(kx)
	fkx = f_vector(kx)
	sign_fr = np.sign(fkx.real)
	sign_fi = np.sign(fkx.imag)

	diff_fr = np.diff(sign_fr)
	diff_fi = np.diff(sign_fi)

	# Zeros must occur when both Real and Imag change sign simultaneously
	cross = np.abs(diff_fr) + np.abs(diff_fi)
	pts = np.where((cross == 4))
	guesses = kx[pts]

	# Find minimum residual near initial guess. Returns OptimizeResult object
	for g in guesses:
		res = sp.optimize.minimize(phi,[g.real,g.imag])
		
		if res.success:
			bx_min, ax_min = res.x
		else:
			print 'Minimize fail.'
			exit()

		kx_min = bx_min + 1j*ax_min

		# Find corresponding Beta (from equation A4)
		B = sqrt(kf**2-kx_min**2)
		print 'kx:   %.3f + %.3fi' % (kx_min.real,kx_min.imag)
		print 'Beta: %.3f + %.3fi' % (B.real,B.imag)

	'''
	# Pretty pictures
	'''
	ext = np.array(putil.getExtent(b,a))
	asp = 'auto'

	fig, ax = plt.subplots(2,2,figsize=(7,7),sharex=True,sharey=True)

	# Residue plot

	ax[0,0].set_title(r'$10\log_{10}(|f|^{2})$')
	
	resIm = ax[0,0].imshow(phiPlotter(kx),extent=ext, vmax=40, vmin=-20, aspect=asp)

	#Colorbar
	# resmag = make_axes_locatable(ax[0])
	# cres = resmag.append_axes("right", size="2%", pad=0.05)
	# cbarres = plt.colorbar(resIm,cax=cres,format='%u')

	ax[0,1].set_title(r'$\Delta$Sign $Re\{f\}$')
	ax[1,1].set_title(r'$\Delta$Sign $Im\{f\}$')
	ax[1,0].set_title(r'Crossings')

	ax[0,1].imshow(diff_fr, extent=ext, aspect=asp, cmap='coolwarm')
	ax[1,1].imshow(diff_fi, extent=ext, aspect=asp, cmap='coolwarm')
	ax[1,0].imshow(cross, extent=ext, aspect=asp, cmap = 'afmhot')

	ax[1,0].set_xlabel(r'$\beta_{x} \lambda_{o}$')
	ax[1,1].set_xlabel(r'$\beta_{x} \lambda_{o}$')
	ax[0,0].set_ylabel(r'$\alpha_{x}\lambda_{o}$')
	ax[1,0].set_ylabel(r'$\alpha_{x}\lambda_{o}$')


	# Beta(n,d,wl=wl,pol=pol,polarity='even',Nmodes=None,plot=True)
	# fig.colorbar(ima)
	plt.show()



	return

if __name__ == '__main__':
  main()
