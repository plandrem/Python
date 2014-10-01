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

	wl = 400
	d = 100

	n = 4 - 10j

	k = 2*pi/wl
	kappa_r = np.linspace(0,abs(n)*k,50)
	kappa_i = np.linspace(0,abs(n)*k/20,50)

	KR,KI = np.meshgrid(kappa_r,kappa_i)

	kappa = KR - 1j*KI

	gamma = lambda x: sqrt((n*k)**2 - x**2 - k**2)
	
	pol = 'TE'
	C = n**2 if pol == 'TM' else 1.

	trans = lambda K: tan(K * d) - C * gamma(K)/K

	T = trans(kappa)

	# T[np.where(np.isinf(T))] = 0

	print T

	ext = np.array(putil.getExtent(kappa_r,kappa_i)) * d/pi
	asp = 'auto'
	
	fig, ax = plt.subplots(3,figsize=(15,9))
	imr = ax[0].imshow(T.real,extent=ext,vmin=-400,vmax=400, aspect=asp)
	imi = ax[1].imshow(T.imag,extent=ext, aspect=asp)
	ima = ax[2].imshow(abs(T),extent=ext, aspect=asp, vmax=100)

	# plt.figure()
	# plt.imshow(abs(tan(kappa*d)))

	# Beta(n,d,wl=wl,pol=pol,polarity='even',Nmodes=None,plot=True)
	fig.colorbar(ima)
	plt.show()



	return

if __name__ == '__main__':
  main()
