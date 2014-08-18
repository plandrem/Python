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

FTIR_PATH = '/Users/Patrick/Documents/PhD/Data/Rectangular Resonator/FTIR/'

def loadReflectanceSpectrum(path,units='wl'):

	data = np.loadtxt(path, delimiter=',')

	wn = data[:,0]
	ref = data[:,1]

	wl = 1/wn * 1e4 #um

	if units == 'wl': return wl,ref
	if units == 'wn': return wn,ref