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

	# EE263 method for polynomial fitting by matrix factorization

	g = lambda x: x**2

	# generate some fake data
	npts = 100
	xs = np.linspace(0,10,npts)
	data = g(xs)

	# add some noise to the "measured" data points
	mu = 0
	sigma = .05

	noise = np.random.normal(mu,sigma,npts)

	data += noise

	# define basis functions
	a = lambda x,n: x**(n)

	# test basis functions
	print 'a(x,0) should be 1:', a(7,0)
	print 'a(x,1) should be x:', a(7,1)
	print 'a(x,2) should be x^2:', a(7,2)
	
	# build basis matrix
	N = 5			# order of polynomial fit (+1)

	a = np.vectorize(a)

	A = np.zeros((npts,N))
	for n in range(N):
		A[:,n] = a(xs,n)

	# Use pseudoinverse to extract optimal coefficients (ie solve y = Ax for x with the smallest norm)
	y = data
	x_ls = np.linalg.lstsq(A,y)[0]

	y_ls = np.dot(A,x_ls)

	# plot results

	print 'Optimal Coefficients:', x_ls

	plt.plot(xs,data,'bo')
	plt.plot(xs,y_ls,'r-')
	plt.show()


	return

if __name__ == '__main__':
  main()
