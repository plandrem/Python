#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

from scipy import exp
from scipy import cosh

from functools import partial

def sech(t):
	return 1/cosh(t)
	
def SechPulse(t,w=1,width=10,chirp=0):
	'''
	returns linearly chirped sech**2 pulse waveform.
	
	w is optical frequency in radial units, [1/fs]
	width is half-width in fs
	FWHM is 2*ln(1 + sqrt(2))*width
	'''
	
	return sech(t/width)*exp(-1j*chirp*t**2/(2*width**2))*exp(-1j*w*t)
	
def Autocorrelation(T,f=None,t=None,Amp=1,mode='I'):
	
	'''
	Perform autocorrelation of function f. t is the primary argument of f, which must include the complete non-zero domain of the function.
	(analytically this should be -inf to inf).  T is tau, the delay.
	
	Mode I returns an intensity autocorrelation (ie. f**2)
	'''
	A = f(t)
	B = f(t-T)
	
	#plt.plot(t,abs(A)**2,'r-',t,abs(B)**2,'b-')
	#plt.show()
	
	if mode == 'I': C = abs((A+B)**2)**2
	else: C = A*B
	
	return Amp*np.trapz(C,x=t)
	
def test_SechPulse():
	t = np.linspace(-60,60,300)
	s = SechPulse(t)
	
	plt.plot(t,s)
	plt.show()

def Sech2AC(T = np.linspace(-300,300,5000),width=50,chirp=1,wl=800):

	'''
	Produces intensity autocorrelation spectrum of chirped hyperbolic secant pulse.
	
	T = tau, delay values of desired autocorrelation signal
	
	width = half width of pulse (fs)
	chirp = linear chirp parameter, 0 = no chirping
	wl = wavelength (nm)
	
	'''

	wl *= 1e-9
	w = 3e8/wl*2*sp.pi * 1e-15
	
	f = partial(SechPulse,w=w,width=width,chirp=chirp)
	
	t = np.linspace(-500,500,5e3)
	
	ac = putil.maplist(Autocorrelation,T,f,t)
	
	
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	
	ax1.plot(t,f(t).real,'r-')
	ax2.plot(T,ac)
	
	ax2.set_xlabel('Delay (fs)')
	ax2.set_ylabel('Intensity (a.u.)')
	plt.show()
	
	
	return ac

def main():
	
	#test_SechPulse()	
	Sech2AC()
	return 0

if __name__ == '__main__':
	main()

