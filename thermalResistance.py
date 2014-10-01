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

	kgst = 0.4

	kal = 300
	dal = 300e-9

	kox = 1
	dox = 10e-6


	w = 500e-9

	dgst = kgst * w/2. * sqrt(dox/kal/kox/dal)
	# dgst = kgst * w * sp.log(8*dal/pi/w)/4/kal

	print dgst

	return

if __name__ == '__main__':
  main()
