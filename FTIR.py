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

prop = mpl.font_manager.FontProperties(fname='/Library/Fonts/GillSans.ttc')

# Change reporting of numpy objects to lower float precision
np.set_printoptions(linewidth=150, precision=3)



FTIR_PATH = '/Users/Patrick/Documents/PhD/Data/Rectangular Resonator/FTIR/'

def loadReflectanceSpectrum(path,units='wl'):

	data = np.loadtxt(path, delimiter=',')

	wn = data[:,0]
	ref = data[:,1]

	wl = 1/wn * 1e4 #um

	if units == 'wl': return wl,ref
	if units == 'wn': return wn,ref

def plotAllFilesInCurrentDirectory(xlim=(1.6,5), ylim=(0,110)):

	data = []
	names = []

	# get list of files in directory
	files = [f for f in os.listdir('.') if os.path.isfile(f)]

	print 'Loading:'
	for f in files:
		name, ext = os.path.splitext(f)
		ext = ext.lower()

		if ext == '.csv':
			print os.path.relpath(f) + '...'
			# load data
			wl,ref = loadReflectanceSpectrum(os.getcwd() + '/' + f)
			data.append((wl,ref))
			names.append(name)

	N = len(data)

	fig, ax = plt.subplots(1,figsize=(5,4))

	for d in data:
		ax.plot(d[0],d[1])

	# plt.title('500x150 nm GST Antenna')
	ax.set_xlabel('Wavelength (nm)',fontproperties=prop)
	ax.set_ylabel('Reflectance (%)',fontproperties=prop)
	ax.legend(names,loc='best')

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	plt.tight_layout()

	plt.show()