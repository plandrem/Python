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

	x = np.linspace(0,5,100)
	y = np.linspace(0,5,100)

	fig, ax = plt.subplots(1,figsize=(3,3))

	ax.plot(x,y)

	plt.xlabel('wavelength',fontsize=14)

	plt.tight_layout()
	plt.show()

	return

if __name__ == '__main__':
  main()
