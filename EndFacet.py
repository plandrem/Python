# !/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

import putil
import sys
import os
import time

# import pudb; pu.db

from DielectricSlab import Beta, numModes

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

np.set_printoptions(linewidth=150, precision=8)

pi = sp.pi
sqrt = sp.emath.sqrt
mu = 4*pi * 1e-7
# c  = 299792458
c = 3e8

# units
cm = 1e-2
um = 1e-6
nm = 1e-9

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

def beta_marcuse(n,d,wl=1.,pol='TM',Nmodes=1,plot=False):

	'''
	Based on theory in Marcuse, 1970 "Radiation Losses in Tapered Dielectric Waveguides"

	Computes beta for dielectric slab in air (can be modified for asymmetric slab)

	Returns array (length Nmodes) of wavevectors - if fewer than Nmodes exist, array is padded 
	with np.nan

	0th index result corresponds to fundamental

	Currently only valid for even modes!

	INPUTS:
	n - index of slab
	d - half-thickness of slab

	'''

	Bs = np.zeros(Nmodes) * np.nan

	k = 2*pi/wl

	K = lambda b: sqrt((n*k)**2 - b**2)
	g = lambda b: sqrt(b**2 - k**2)

	C = n**2 if pol == 'TM' else 1.

	trans = lambda b: tan(K(b) * d) - C * np.real(g(b)/K(b))

	# Find zero crossings, then use the brentq method to find the precise zero crossing 
	# between the two nearest points in the array

	b = np.linspace(0,n*k - (1e-15),10000)
	diff = np.diff(np.sign(trans(b)))

	for i,idx in enumerate(np.nonzero(diff)[0][::-1]):

		b_low = b[idx-1]
		b_high = b[idx+1]

		if i < Nmodes: Bs[i] = sp.optimize.brentq(trans,b_low,b_high)

	if plot:
		plt.plot(b*d,tan(K(b)*d),'r')
		plt.plot(b*d,C * np.real(g(b)/K(b)),'b')
		plt.plot(b*d,trans(b),'k')
		plt.plot(b*d,np.sign(trans(b)),'m:')
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

def pythonic_main():
	'''
	rewrite of the algorithm with 3 goals: 1) improve performance by using numpy methods instead
	of nested for-loops; 2) catch/eliminate typos from the first attempt; and 3) build a more
	elegant API so I'm not recalculating k in every function call.
	'''

	# Define Key Simulation Parameters

	n = sqrt(20)
	d = 1 * cm
	kd = np.array([0.209,0.418,0.628,0.837,1.04,1.25])
	pol = 'TE'

	p_max = 20 			# max transverse wavevector to use for indefinite integrals; multiples of k
	p_res = 2e3     # number of steps to use in integration

	imax = 4 				# max number of iterations to run

	'''
	Calculate wavevectors and other physical quantities needed for all functions
	'''

	# constants
	k = kd/d
	wl = 2*pi/k
	w = c*k
	eps = n**2

	
	# # Debugger for mode solver
	# wl = 1.0
	# k = 2*pi/wl
	# kd = 1.0
	# d = kd/k	
	# print beta_marcuse(1.432,d,wl=wl,pol='TE',plot=True) * d
	# print 'Should be 1.21972'
	# exit()
	

	# wavevectors
	N = numModes(n,1,2*d/wl)
	Bm = beta_marcuse(n,d,wl=wl,Nmodes=N, pol=pol)			# Propagation constants of waveguide modes
	Bo = Bm[0,:]

	print Bm.shape()
	exit()

	p = np.linspace(1e-15,p_max*k,p_res)					# transverse wavevectors in air (independent variable for algorithm)
	dp = p[1]-p[0]
	
	Bc = sqrt(       k**2 - p**2) 						# propagation constants in air for continuum modes
	o  = sqrt(n**2 * k**2 - Bc**2)						# transverse wavevectors in high-index region for continuum modes
	
	gm  = sqrt(      -k**2 + Bm**2)						# transverse wavevectors in air for guided modes
	Km  = sqrt(n**2 * k**2 - Bm**2)						# transverse wavevectors in high-index region for guided modes

	Bc = Bc.real - 1j*Bc.imag
	o =  o.real  - 1j*o.imag
	gm = gm.real - 1j*gm.imag
	Km = Km.real - 1j*Km.imag

	



	# Mode Amplitude Coefficients
	P = 1

	Am = sqrt(2*w*mu*P / (Bm*d + Bm/gm))

	# Bt = sqrt(2*w*mu*P / (pi*Bc)) # might need to make these |Bc| instead of Bc
	# Br = sqrt(2*p**2*w*mu*P / (pi*Bc*(p**2*cos(o*d)**2 + o**2*sin(o*d)**2)))

	Bt = sqrt(2*w*mu*P / (pi*abs(Bc))) # might need to make these |Bc| instead of Bc
	Br = sqrt(2*p**2*w*mu*P / (pi*abs(Bc)*(p**2*cos(o*d)**2 + o**2*sin(o*d)**2)))
	
	Dr = 1/2. * exp(-1j*p*d) * (cos(o*d) + 1j*o/p * sin(o*d))

	# plt.plot(p/k,Bt,'r')
	# plt.plot(p/k,Br.real,'b')
	# plt.show()
	# print Am
	# exit()






	# Overlap Integral Solutions, Gm(p) and F(p',p) = F(p1,p2)

	G = np.zeros((N,p_res),dtype = 'complex')

	for i in range(N):
		G[i,:] = 2 * k**2 * (eps-1) * Am[i] * Bt * cos(Km[i]*d) * (gm[i]*cos(p*d) - p*sin(p*d)) / ((Km[i]**2 - p**2)*(gm[i]**2 + p**2))

	p1,p2 = np.meshgrid(p,p)			# Move to 2D arrays
	Br1 = np.tile(Br,(p_res,1))
	Bt1 = np.tile(Bt,(p_res,1)); Bt2 = Bt1.transpose()
	o1  = np.tile(o ,(p_res,1))
	Bc1 = np.tile(Bc,(p_res,1)); Bc2 = Bc1.transpose()
	
	D1  = np.tile(Dr,(p_res,1))
	Dstar  = np.conjugate(D1)

	od = o1*d; pd = p2*d

	# This method for F is based on a more direct solution of the field overlap integral, with less
	# subsequent algebra:
	# F  = 2*Br1*Bt2 * ((o1*sin(od)*cos(pd) - p2*cos(od)*sin(pd))/(o1**2-p2**2) + \
	# 								 2*(D1 * (exp(1j*p1*d) * (p2*sin(pd)+1j*p1*cos(pd))) - 1j*p1 ).real / (p1**2-p2**2) )
	
	# Handle Cauchy singularities
	cauchy = pi * Bt1.transpose() * Br1 * (D1 + Dstar)
	idx = np.where(1 - np.isfinite(F))
	# F[idx] = 0
	# F[idx] = cauchy[idx]

	# Gelin's definition of F, with the sin arguments corrected:
	F = np.eye(p_res) * Bt1.transpose() * Br1 * (D1 + Dstar) - \
			np.nan_to_num(k**2 * (eps-1) * Bt2 * Br1 * (sin((o1+p2)*d)/(o1+p2) + sin((o1-p2)*d)/(o1-p2)) / (p1**2-p2**2)) * (1-np.eye(p_res))




	'''
	Run Neuman series and test for convergence of the fields
	'''

	# Initialize coefficients to zero
	am = np.zeros(N)
	qr = np.zeros(p_res)	

	for i in range(imax):

		print '\nComputing iteration %u of %u' % (i+1,imax)

		# Qt
		qr1 = np.tile(qr,(p_res,1))
		integral = np.trapz(qr1 * (Bo-Bc1) * F, dx=dp, axis=1)
		# integral = np.sum(qr1 * (Bo-Bc1) * F, axis=1) * dp

		sigma = np.sum([(Bo-Bm[n]) * am[n] * G[n,:] for n in range(N)], axis=0)

		qt = 1/(2*w*mu*P) * abs(Bc) / (Bo+Bc) * (2*Bo*G[0,:] + integral + sigma)

		# an
		am_prev = am
		am = np.array(
			[ (1/(4*w*mu*P) * np.sum(qt * (Bm[n]-Bc) * G[n,:]) * dp) for n in range(N) ]
			)
		print am
		print 'Delta ao: ', abs(am_prev-am)


		#Qr
		qt1 = np.tile(qt,(p_res,1))
		integral = np.trapz(qt1 * (Bc2-Bc1) * F.transpose(), dx=dp, axis=1)
		# integral = np.sum(qt1 * (Bc2-Bc1) * F.transpose(), axis=1) * dp
		
		qr = 1/(4*w*mu*P) * (abs(Bc)/Bc) * integral

		'''
		Test Power Conservation
		'''

		error = (1+am[0]) * (1-am[0].conjugate()) - np.sum(abs(am[1:])**2) - np.sum((abs(qt)**2 + abs(qr)**2) * np.conjugate(Bc) / abs(Bc)) * dp
		print 'Power Conservation: ', abs(error.real)

		'''
		Test Gelin, eq. 14
		'''

		print 1/(4*w*mu*P) * np.trapz(qt * (Bm[0]+Bc) * G[0,:], dx=dp)

	'''
	Plot results
	'''

	fig, ax = plt.subplots(2,figsize=(7,5),sharex=True)

	ax[0].plot(p/k,qt.real,'r')
	ax[0].plot(p/k,qt.imag,'r:')
	ax[1].plot(p/k,qr.real,'b')
	ax[1].plot(p/k,qr.imag,'b:')

	ax[0].axhline(0,color='k',ls=':')
	ax[1].axhline(0,color='k',ls=':')

	ax[0].set_xlim(0,150/k)
	# ax[0].set_xlim(0,1,5)
	# fig2, ax2 = plt.subplots(1,figsize=(7,5))
	# ax2.plot(p/k,Bc.real,'g')
	# ax2.plot(p/k,Bc.imag,'g:')

	# plt.figure()
	# ext = putil.getExtent(p/k,p/k)
	# plt.imshow(np.real(F), vmin=-1e14, vmax=1e14, extent = ext)
	# plt.colorbar()

	# plt.figure()
	# ext = putil.getExtent(p/k,p/k)
	# plt.imshow(abs(Bc2-Bc1), extent = ext)
	# plt.colorbar()

	# plt.figure()
	# ext = putil.getExtent(p/k,p/k)
	# plt.imshow(abs(qt1), extent = ext)
	# plt.colorbar()


	plt.show()



def main():

	'''
	Reminder: 'd' in Marcuse notation is the half height of the slab, whereas functions from DielectricSlab
	refer to 'd' as height / wavelength. Below, we use Marcuse notation to avoid confusion with references.
	'''
	
	# Define Waveguide Constants
	n  = sqrt(20)
	d = 1 * cm
	kd = 0.628
	k = kd/d

	wl = 2*pi/k
	w = c*k

	# for d in ds:

	# get all waveguide modes supported by structure
	N = numModes(n,1,2*d/wl)
	B_modes = beta_marcuse(n,d,wl=wl,pol='TE',Nmodes=N) # propagation constant
	Bo = B_modes[0]

	print n, d, wl, N
	print B_modes

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
	ps = np.linspace(1e-15,20*k,100)			# must be linear for integration
	if np.where(ps==k):								# avoid singularity at p = k (triggers B = 0 in denominator of Be)
		ps[np.where(ps==k)] += 1e-15

	# Initialize coefficients to zero
	am_i = np.zeros(N)
	qr_i = np.zeros(len(ps))

	imax = 4
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

	print '\nTesting Power Conservation:'
	dp = ps[1]-ps[0]
	Bc = lambda x: B_continuum(x,k)

	sigma = np.sum(abs(B_modes[1:])**2)
	integral = np.sum(abs(qt_i)**2 + abs(qr_i)**2 * np.conjugate(Bc(p))/abs(Bc(p))) * dp
	error = np.real((1+am_i[0]) * (1-np.conjugate(am_i[0])) - integral - sigma)
	print 'error:', error
	
	
	fig, ax = plt.subplots(2,figsize=(7,5),sharex=True)

	ax[0].plot(ps,qt_i.real,'r')
	ax[0].plot(ps,qt_i.imag,'r:')
	ax[1].plot(ps,qr_i.real,'b')
	ax[1].plot(ps,qr_i.imag,'b:')

	ax[0].axhline(0,color='k',ls=':')
	ax[1].axhline(0,color='k',ls=':')

	fig2, ax2 = plt.subplots(2,figsize=(7,5),sharex=True)

	ax2[0].plot(ps_f,qt_f.real,'r')
	ax2[0].plot(ps_f,qt_f.imag,'r:')
	ax2[1].plot(ps_f,qr_f.real,'b')
	ax2[1].plot(ps_f,qr_f.imag,'b:')

	ax2[0].axhline(0,color='k',ls=':')
	ax2[1].axhline(0,color='k',ls=':')

	# plt.figure()
	# plt.plot(ps_f,B_continuum(ps_f,k).real,'r')
	# plt.plot(ps_f,B_continuum(ps_f,k).imag,'b')

	# plt.figure()
	# plt.plot(ps_f,Bt(w,ps_f).real,'r')
	# plt.plot(ps_f,Bt(w,ps_f).imag,'b')

	# plt.figure()
	# plt.plot(ps_f,Br(w,n,d,ps_f).real,'r')
	# plt.plot(ps_f,Br(w,n,d,ps_f).imag,'b')
	
	# plt.figure()
	# plt.plot(ps_f,o(n,ps_f,k).real,'r')
	# plt.plot(ps_f,o(n,ps_f,k).imag,'b')

	# ax[1].set_xlim(0,150)
	
	

	plt.show()


'''
Helper Functions
'''

def B_continuum(p,k):
	Bc = sqrt(k**2-p**2)
	return Bc.real - 1j * Bc.imag

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
		# return pi * Bt(w,p) * Br(w,n,d,p2) * (Dr(p2,w,n,d) + np.conjugate(Dr(p2,w,n,d)))
		return 0

	else:
		return 2*Br(w,n,d,p2) * Bt(w,p) * ((s2*sin(s2*d)*cos(p*d) - p*cos(s2*d)*sin(p*d))/(s2**2-p**2) + \
									 2*(Dr(p2,w,n,d) * (exp(1j*p2*d) * (p *sin(p*d)+1j*p2*cos(p*d))) - 1j*p2 ).real / (p2**2-p**2) )


def QT(p,w,n,d,B_modes,am,qr,ps):
	
	P = 1
	k = w/c

	Bo = B_modes[0]
	Bc = lambda p: B_continuum(p,k)
	Gm = lambda m,p: G(m,p,w,n,d,B_modes[m])

	Fpp2 = [F(p2,p,w,n,d) for p2 in ps]

	integrand = qr * (Bo - Bc(ps)) * Fpp2
	dp = ps[1]-ps[0]
	integral = np.sum(integrand * dp)

	sigma = np.sum(np.array([(Bo-B_modes[m])*am[m]*Gm(m,p) for m in range(len(B_modes))]))

	return 1/(2*w*mu*P) * abs(Bc(p)) / (Bo + Bc(p)) * (2*Bo*Gm(0,p) + integral + sigma)

def AM(m,w,n,d,B_modes,qt,ps):

	P = 1

	k = w/c

	Bm = B_modes[m]
	Bc = lambda p: B_continuum(p,k)
	Gm = lambda p: G(m,p,w,n,d,Bm)

	integrand = qt * (Bm - Bc(ps)) * Gm(ps)
	dp = ps[1]-ps[0]
	integral = np.sum(integrand * dp)


	return 1/(4*w*mu*P) * integral

def QR(p,w,n,d,qt,ps):

	P = 1

	k = w/c
	if p == k: return 0

	Bc = lambda p2: B_continuum(p2,k)

	Fpp2 = [F(p,p2,w,n,d) for p2 in ps]

	integrand = qt * (Bc(p) - Bc(ps)) * Fpp2
	dp = ps[1]-ps[0]
	integral = np.sum(integrand * dp)

	return 1/(4*w*mu*P) * abs(Bc(p))/Bc(p) * integral



if __name__ == '__main__':
  # test_beta()
  # test_beta_marcuse()
  pythonic_main()
  # main()
