#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

def Gauss(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def GaussDC(x,A,mu,sigma,DC):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + DC

def SimpleGaussianFit(xs,ys):
	
	x = sum(xs*ys)/sum(ys)
	width = sp.sqrt(abs(sum((xs-x)**2*ys)/sum(ys)))
	 
	max = ys.max()
	fit = lambda t : max*sp.exp(-(t-x)**2/(2*width**2))
	
	gaussian = fit(xs)
	
	plt.plot(xs,gaussian,'r-')
	plt.plot(xs,ys,'bo')
	
	plt.show()
	
def GaussianFit(xs,ys,guess=None):
	# good initial parameters guess for lab = (8,0.002,0.001,0.2)
	
	popt = curve_fit(GaussDC,xs,ys,guess)[0]
	
	#A,mu,sigma = popt
	A,mu,sigma,DC = popt
	sigma = abs(sigma)
	#result = Gauss(xs,A,mu,sigma)
	result = GaussDC(xs,A,mu,sigma,DC)
	
	
	plt.figure()
	plt.plot(xs,result,'r-')
	plt.plot(xs[::2],ys[::2],'bo')
	plt.xlabel('Time (s)')
	plt.ylabel('V')
	
	
	return A,mu,sigma,DC
	
def Waist(x,wo,xo=0):
	l = 632.8e-9
	xr = sp.pi*wo**2/l    #Rayleigh length
	return wo * sp.sqrt(1 + ((x-xo)/xr)**2)
	
def WaistMap(xs,wo,xo=0):
	return map(Waist,xs,np.ones(len(xs))*wo,np.ones(len(xs))*xo)
	
def WaistFit(xs,ys,guess=(1e-4,-1)):
	return curve_fit(Waist,xs,ys,guess)[0]

def WaistTest():
	xs = np.linspace(-10e-2,0.3,5000)
	wo = 1e-3
	xo = 0.1
	ys = WaistMap(xs,wo,xo)
	
	plt.plot(xs,ys)
	plt.show()
	exit()
	
def main():
	
	return 0

if __name__ == '__main__':
	main()

