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

	income = 1e5
	cash = [income]

	for i in range(10):
		cash.append(cash[i]*1.04 + income)

	cash = np.array(cash)
	plt.plot(cash/1e6)
	plt.axhline(1, color='k',ls=':')
	plt.show()

	return

if __name__ == '__main__':
  main()
