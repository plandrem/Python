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
	d = 1e-2
	kd = 0.628
	k = kd/d

	wl = 2*pi/k
	w = c*k

	# for d in ds:

	# get all waveguide modes supported by structure
	N = numModes(n,1,2*d/wl)
	B_modes = beta_marcuse(n,d,wl=wl,pol='TE',Nmodes=N) # propagation constant
	Bo = B_modes[0]

	'''
	# Debugger for mode solver
	wl = 2.0
	k = 2*pi/wl
	kd = 1.0
	d = kd/k	
	print beta_marcuse(1.432,d,wl=wl,pol='TE') * d
	exit()
	'''

	# transverse wavevectors of continuum modes outside structure
	ps = np.linspace(0,20*k,50)			# must be linear for integration
	if np.where(ps==k):								# avoid singularity at p = k (triggers B = 0 in denominator of Be)
		ps[np.where(ps==k)] += 1e-15


	# Initialize coefficients to zero
	am_i = np.zeros(N)
	qr_i = np.zeros(len(ps))

	imax = 1
	for i in range(imax):

		print '\nComputing iteration %u of %u' % (i+1,imax)

		qt_i = np.array([QT(p,w,n,d,B_modes,am_i,qr_i,ps) for p in ps])
		
		am_i = np.array([AM(m,w,n,d,B_modes,qt_i,ps) for m in range(N)])
		print am_i

		qr_i = np.array([QR(p,w,n,d,qt_i,ps) for p in ps])

	print '\nComputing q values for plot...'

	ps_f = np.linspace(0,150,100)
	qt_f = np.array([QT(p,w,n,d,B_modes,am_i,qr_i,ps) for p in ps_f])
	qr_f = np.array([QR(p,w,n,d,qt_i,ps) for p in ps_f])
	

	fig, ax = plt.subplots(2,figsize=(7,5),sharex=True)

	ax[0].plot(ps_f,qt_f.real,'r')
	ax[0].plot(ps_f,qt_f.imag,'r:')
	ax[1].plot(ps_f,qr_f.real,'b')
	ax[1].plot(ps_f,qr_f.imag,'b:')

	ax[0].axhline(0,color='k',ls=':')
	ax[1].axhline(0,color='k',ls=':')

	plt.figure()
	plt.plot(ps_f,B_continuum(ps_f,k).real,'r')
	plt.plot(ps_f,B_continuum(ps_f,k).imag,'b')

	# ax[1].set_xlim(0,150)
	
	

	plt.show()


'''
Helper Functions
'''

def B_continuum(p,k):
	return sqrt(k**2-p**2)

def o(n,p,k):
	# transverse wavevector inside structure for continuum modes

	B = B_continuum(p,k)
	return sqrt((n*k)**2 - B**2)

def g(B,k):
	# transverse wavevector outside structure for guided modes
	return sqrt(B**2 - k**2)

def K(n,B,k):
	# transverse wavevector inside structure for guided modes
	return sqrt((n*k)**2 - B**2)	

def A(w,B,d,g):
	# Amplitude Coefficient of guided modes
	P = 1
	return sqrt(2*w*mu*P / (B*d + B/g))

def Br(w,n,d,p):
	# Amplitude Coefficient of continuum modes outside structure
	P = 1
	k = w/c
	s = o(n,p,k)
	B = B_continuum(p,k)

	return sqrt(2*p**2*w*mu*P / (pi*abs(B)*(p**2*cos(s*d)**2 + s**2*sin(s*d)**2)))

def Bt(w,p):
	# Amplitude Coefficient of continuum modes in free space

	k = w/c
	P = 1
	B = B_continuum(p,k)

	return sqrt(2*P*w*mu / (pi*abs(B)))

def Dr(p,w,n,d):
	k = w/c
	s = o(n,p,k)

	if p == 0: return 0.5 * exp(-1j*p*d) * cos(s*d) 

	return 0.5 * exp(-1j*p*d) * (cos(s*d) + 1j*s/p * sin(s*d))

def G(m,p,w,n,d,Bm):
	k = w/c

	Km = K(n,Bm,k)
	gm = g(Bm,k)
	Am = A(w,Bm,d,gm)

	return 2 * k**2 * (n**2-1) * Am * Bt(w,p) * cos(Km*d) * (gm*cos(p*d) - p*sin(p*d)) / (Km**2-p**2) / (gm**2 + p**2)

def F(p2,p,w,n,d):
	k = w/c

	s = o(n,p,k)
	s2 = o(n,p2,k)

	if p == p2:
		return pi * Bt(w,p) * Br(w,n,d,p2) * (Dr(p2,w,n,d) + np.conjugate(Dr(p2,w,n,d)))

	else:
		return k**2 * (sqrt(n)-1) * Br(w,n,d,p2) * Bt(w,p) / (p2**2 - p**2) * ( (p*cos(s2*d)*sin(p*d) - s2*sin(s2*d)*cos(p*d))/(s2**2 - p**2) ) 
		# return -k**2 * (sqrt(n)-1) * Br(w,n,d,p2) * Bt(w,p) * (sin((s2+p)*d)/(s2+p) + sin((s2-p)*d)/(s2-p)) / (p2**2 - p**2)


def QT(p,w,n,d,B_modes,am,qr,ps):
	
	P = 1
	k = w/c

	Bo = B_modes[0]
	Bc = lambda p: B_continuum(p,k)
	Gm = lambda m,p: G(m,p,w,n,d,B_modes[m])

	Fpp2 = [F(p2,p,w,n,d) for p2 in ps]

	integrand = qr * (Bo - Bc(ps)) * Fpp2
	dp = ps[1]-ps[0]
	integral = sum(integrand * dp)

	sigma = sum(np.array([(Bo-B_modes[m])*am[m]*Gm(m,p) for m in range(len(B_modes))]))

	return 1/(2*w*mu*P) * abs(Bc(p)) / (Bo + Bc(p)) * (2*Bo*Gm(0,p) + integral + sigma)

def AM(m,w,n,d,B_modes,qt,ps):

	P = 1

	k = w/c

	Bm = B_modes[m]
	Bc = lambda p: B_continuum(p,k)
	Gm = lambda p: G(m,p,w,n,d,Bm)

	integrand = qt * (Bm - Bc(ps)) * Gm(ps)
	dp = ps[1]-ps[0]
	integral = sum(integrand * dp)


	return 1/(4*w*mu*P) * integral

def QR(p,w,n,d,qt,ps):

	P = 1

	k = w/c
	if p == k: return 0

	Bc = lambda p2: B_continuum(p2,k)

	Fpp2 = [F(p,p2,w,n,d) for p2 in ps]

	integrand = qt * (Bc(p) - Bc(ps)) * Fpp2
	dp = ps[1]-ps[0]
	integral = sum(integrand * dp)


	return 1/(4*w*mu*P) * abs(Bc(p))/Bc(p) * integral

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
