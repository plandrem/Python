#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

import putil
import sys
import os

from DielectricSlab import Beta, numModes

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt
mu = 4*pi * 1e7
c  = 299792458

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

def beta_marcuse(n,d,wl=1,pol='TM',Nmodes=1,plot=False):

	'''
	Based on theory in Marcuse, 1970 "Radiation Losses in Tapered Dielectric Waveguides"

	Computes beta for dielectric slab in air (can be modified for asymmetric slab)

	Returns array (length Nmodes) of wavevectors - if fewer than Nmodes exist, array is padded 
	with np.nan

	0th index result corresponds to fundamental

	INPUTS:
	n - index of slab
	d - half-thickness of slab

	'''

	Bs = np.zeros(Nmodes) * np.nan

	k = 2*pi/wl

	K = lambda b: sqrt((n*k)**2 - b**2)
	g = lambda b: sqrt(b**2 - k**2)

	C = n**2 if pol == 'TM' else 1

	trans = lambda b: tan(K(b) * d) - C * np.real(g(b)/K(b))

	b = np.linspace(0,n*k - (1e-15),10000)
	diff = np.diff(np.sign(trans(b)))

	for i,idx in enumerate(np.nonzero(diff)[0][::-1]):

		b_low = b[idx-1]
		b_high = b[idx+1]

		if i < Nmodes: Bs[i] = sp.optimize.brentq(trans,b_low,b_high)

	if plot:
		plt.plot(b,tan(K(b)*d),'r')
		plt.plot(b,C * np.real(g(b)/K(b)),'b')
		plt.plot(b,trans(b),'k')
		plt.plot(b,np.sign(trans(b)),'m:')
		plt.ylim(-10,10)
		plt.show()

	return Bs

def test_beta_marcuse():

	target_vals_TE = [0.25781,0.54916,1.21972,1.93825,2.66839,4.13075]
	target_vals_TM = [0.25207,0.51677,1.12809,1.84210,2.58934,4.08131]

	wl = 1

	n = 1.432
	k = 2*pi/wl

	kds = np.array([0.25,0.5,1.0,1.5,2.0,3.0])
	ds = kds/k

	print 'TE'
	for i,d in enumerate(ds):
		bTE = beta_marcuse(n,d,pol='TE',wl=wl)[0]
		print "%.5f, %.5f, %.5f" % (bTE * d, target_vals_TE[i], abs((bTE * d - target_vals_TE[i]) / target_vals_TE[i]))

	print
	print 'TM'

	for i,d in enumerate(ds):
		bTM = beta_marcuse(n,d,pol='TM',wl=wl)[0]
		print "%.5f, %.5f, %.5f" % (bTM * d, target_vals_TM[i], abs((bTM * d - target_vals_TM[i]) / target_vals_TM[i]))


def test_beta():

	'''
	Compare mode solver solutions against tabulated values in Marcuse, 1970 
	"Radiation Losses in Tapered Dielectric Waveguides"

	To avoid confusion, Marcuse's 'd' (which is half the slab height) will be referred to as 'a'
	d here is h/wl = 2*a/wl
	'''

	target_vals = [0.25781,0.54916,1.21972,1.93825,2.66839,4.13075]

	wl = 1.0
	k = 2*pi/wl


	print "TE\n"

	kas = np.array([0.25,0.5,1.0,1.5,2.0,3.0])

	for i,ka in enumerate(kas):
		a = ka/k

		h = 2 * a

		d = h/wl

		b = Beta(1.432,1.0,d,1,pol='TE')[0] * wl

		print "%.5f, %.5f, %.5f" % (b*a, target_vals[i], abs((b*a - target_vals[i]) / target_vals[i]))

	print 
	print "TM\n"

	target_vals = [0.25207,0.51677,1.12809,1.84210,2.58934,4.08131]

	for i,ka in enumerate(kas):
		a = ka/k

		h = 2 * a

		d = h/wl

		b = Beta(1.432,1,d,1,pol='TM')[0] * wl

		print "%.5f, %.5f, %.5f" % (b*a, target_vals[i], abs((b*a - target_vals[i]) / target_vals[i]))



def main():

	'''
	Reminder: 'd' in Marcuse notation is the half height of the slab, whereas functions from DielectricSlab
	refer to 'd' as height / wavelength. Below, we use Marcuse notation to avoid confusion with references.
	'''
	
	# Define Waveguide Constants
	n  = sqrt(20)
	wl = 1

	kd = 0.628

	k = 2*pi/wl
	w = c*k

	d = kd/k

	# for d in ds:

	# get all waveguide modes supported by structure
	N = numModes(n,1,2*d/wl)
	B_modes = beta_marcuse(n,d,pol='TE',Nmodes=N) # propagation constant
	Bo = B_modes[0]

	# Initialize coefficients to zero
	an = np.zeros(N)
	qr = 0

	for i in range(100):

		qt_i = lambda p: qt(p,w,n,d,B_modes,an,qr)

	ps = np.linspace(0,1.5,100)
	qts = np.array([qt_i(x) for x in ps])

	plt.plot(ps,qts.real,'r')
	plt.show()


'''
Helper Functions
'''

def B_continuum(p,k):
	return sqrt(k**2-p**2)

def o(n,p,k):
	# transverse wavevector inside structure for continuum modes
	return sqrt((n*k)**2 - p**2)

def g(B,k):
	# transverse wavevector outside structure for guided modes
	return sqrt(B**2 - k**2)

def K(n,B,k):
	# transverse wavevector inside structure for guided modes
	return sqrt(B**2 - (n*k)**2)	

def A(w,B,d,g):
	# Amplitude Coefficient of guided modes
	P = 1
	return sqrt(2*w*mu*P / (B*d + B/g))

def Be(w,n,d,p):
	# Amplitude Coefficient of continuum modes outside structure
	P = 1
	k = w/c
	s = o(n,p,k)
	B = B_continuum(p,k)

	return sqrt(2*p**2*w*mu*P / (pi*B*(p**2*cos(s*d)**2 + s**2*sin(s*d)**2)))

def G(m,p,w,n,d,Bm):
	k = w/c

	Km = K(n,Bm,k)
	gm = g(Bm,k)
	Am = A(w,Bm,d,gm)

	return 2 * k**2 * (n**2-1) * Am * Be(w,n,B,d,p) * cos(Km*d) * (gm*cos(p*d) - p*sin(p*d) / (Km**2-p**2) / (gm**2 + p**2))

def F(p,p2,w,n,d):
	k = w/c

	s = o(n,p,k)
	s2 = o(n,p2,k)

	if p == p2:
		return 0

	else:
		return -k**2 * (sqrt(n)-1) * Be(w,n,d,p) * Be(w,n,d,p2) * (sin(s2+p)*d/(s+p) + sin(s2-p)/(s2-p)) / (p2**2 - p**2)


def qt(p,w,n,d,B_modes,an,qr):
	P = 1
	k = w/c
	Bo = B_modes[0]
	Bc = B_continuum(p,k)
	Go = lambda p: G(0,p,w,n,d,Bo)

	return 1/(2*w*mu*P) * abs(Bc) / (Bo + Bc) * (2*Bo*Go(p))

def test_qt():

	wl = 1
	k = 2*pi/wl
	w = c*k

	n = sqrt(20)
	d = 0.5/k

	Bo = beta_marcuse(n,d,wl=wl,pol='TE',Nmodes=1)[0]

	p = np.linspace(0,1.5,100)

	qts = np.array([qt(x,w,n,d,Bo) for x in p])

	plt.plot(p,qts.real,'r')
	plt.plot(p,qts.imag,'b')

	plt.show()


if __name__ == '__main__':
  # test_beta()
  # test_beta_marcuse()
  # test_qt()
  main()
