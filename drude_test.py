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


def main():

	wls = np.linspace(500,2000,100)

	fig, axs = plt.subplots(2,sharex=True,figsize=(5.5,7))

	eps = putil.getEps('cu',wls)
	eps_drude = putil.getEpsDrude('Cu',wls)

	axs[0].plot(wls,eps.real,c='r',lw=2)
	axs[0].plot(wls,eps_drude.real,c='r',lw=2,ls=':')

	axs[1].plot(wls,eps.imag,c='r',lw=2)
	axs[1].plot(wls,eps_drude.imag,c='r',lw=2,ls=':')

	axs[0].set_xlim(np.amin(wls),np.amax(wls))

	axs[1].set_xlabel('Wavelength (nm)')
	axs[0].set_ylabel(r'Re{$\epsilon$}')
	axs[1].set_ylabel(r'Im{$\epsilon$}')

	plt.tight_layout()
	plt.show()

	return

if __name__ == '__main__':
  main()
