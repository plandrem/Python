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
c  = 299792458

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

	# if type(d) not in [int, float] and len(d) > 1:
	# 	d = d.reshape(len(d),1)
	# 	b = np.tile(b,(len(d),1))

	diff = np.diff(np.sign(trans(b)))

	# plt.plot(b[0,:],trans(b)[0,:])
	# plt.plot(b[0,:],np.sign(trans(b))[0,:])
	# plt.ylim(-2,2)
	# plt.show()
	# exit()

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

	return np.array(Bs)

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

def main():
	'''
	rewrite of the algorithm with 3 goals: 1) improve performance by using numpy methods instead
	of nested for-loops; 2) catch/eliminate typos from the first attempt; and 3) build a more
	elegant API so I'm not recalculating k in every function call.
	'''

	# Define Key Simulation Parameters

	n = sqrt(20)
	wl = 10. # Set to 10 to match Gelin values for qt
	kd = np.array([0.209,0.628])
	# kd = np.array([0.209,0.418,0.628,0.837,1.04,1.25])
	# kd = np.linspace(1e-15,1.5,100)
	pol = 'TE'

	p_max = 20 			# max transverse wavevector to use for indefinite integrals; multiples of k
	p_res = 1e3     # number of steps to use in integration

	imax = 4 				# max number of iterations to run

	'''
	Calculate wavevectors and other physical quantities needed for all functions
	'''

	# constants
	k = 2*pi/wl
	d = kd/k
	w = c*k
	eps = n**2

	Nds = len(d)
	
	# # Debugger for mode solver
	# wl = 1.0
	# k = 2*pi/wl
	# kd = 1.0
	# d = kd/k	
	# print beta_marcuse(1.432,d,wl=wl,pol='TE',plot=True) * d
	# print 'Should be 1.21972'
	# exit()
	

	# wavevectors

	Ns = numModes(n,1,2*d/wl)
	N = np.amax(Ns)
	
	Bo = np.zeros(len(d))
	Bm = np.zeros((np.amax(N),len(d)))

	for j,dj in enumerate(d):
		Bm[:,j] = beta_marcuse(n,dj,wl=wl,Nmodes=N, pol=pol)			# Propagation constants of waveguide modes
		Bo[j] = Bm[0,j]

	p = np.tile(np.linspace(1e-15,p_max*k,p_res), (len(d),1))					# transverse wavevectors in air (independent variable for algorithm)
	p = p.transpose()

	dp = p[1,0]-p[0,0]
	
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

	Bt = sqrt(2*w*mu*P / (pi*abs(Bc))) 
	Br = sqrt(2*p**2*w*mu*P / (pi*abs(Bc)*(p**2*cos(o*d)**2 + o**2*sin(o*d)**2)))
	
	Dr = 1/2. * exp(-1j*p*d) * (cos(o*d) + 1j*o/p * sin(o*d))

	# plt.plot(p/k,Bt,'r')
	# plt.plot(p/k,Br.real,'b')
	# plt.show()
	# print Am
	# exit()






	# Overlap Integral Solutions, Gm(p) and F(p',p) = F(p1,p2)

	G = np.zeros((N,p_res,Nds),dtype = 'complex')

	for i in range(N):
		G[i,:,:] = 2 * k**2 * (eps-1) * Am[i] * Bt * cos(Km[i]*d) * (gm[i]*cos(p*d) - p*sin(p*d)) / ((Km[i]**2 - p**2)*(gm[i]**2 + p**2))

	# Convert to multi-dimensional arrays to work with functions of p',p. When transposing, 
	# the index refering to d is kept as the final index

	p1  = np.tile(p, (p_res,1,1)); p2 = p1.transpose(1,0,2)
	Br1 = np.tile(Br,(p_res,1,1))
	Bt1 = np.tile(Bt,(p_res,1,1)); Bt2 = Bt1.transpose(1,0,2)
	o1  = np.tile(o ,(p_res,1,1))
	Bc1 = np.tile(Bc,(p_res,1,1)); Bc2 = Bc1.transpose(1,0,2)
	
	D1  = np.tile(Dr,(p_res,1,1))
	Dstar  = np.conjugate(D1)

	od = o1*d; pd = p2*d.transpose()

	# This method for F is based on a more direct solution of the field overlap integral, with less
	# subsequent algebra:
	# F  = 2*Br1*Bt2 * ((o1*sin(od)*cos(pd) - p2*cos(od)*sin(pd))/(o1**2-p2**2) + \
	# 								 2*(D1 * (exp(1j*p1*d) * (p2*sin(pd)+1j*p1*cos(pd))) - 1j*p1 ).real / (p1**2-p2**2) )
	
	# Handle Cauchy singularities
	# cauchy = pi * Bt2 * Br1 * (D1 + Dstar)
	# idx = np.where(1 - np.isfinite(F))
	# F[idx] = 0
	# F[idx] = cauchy[idx]

	# Gelin's definition of F, with the sin arguments corrected:
	I = np.tile(np.eye(p_res), (Nds,1,1)).transpose(1,2,0)
	F = I * Bt2 * Br1 * (D1 + Dstar) - \
			np.nan_to_num(k**2 * (eps-1) * Bt2 * Br1 * (sin((o1+p2)*d)/(o1+p2) + sin((o1-p2)*d)/(o1-p2)) / (p1**2-p2**2)) * (1-I)





	'''
	Run Neuman series and test for convergence of the fields
	'''

	# Initialize coefficients to zero
	am = np.zeros((N,Nds))
	qr = np.zeros((p_res,Nds))	

	for i in range(imax):

		print '\nComputing iteration %u of %u' % (i+1,imax)

		# Qt
		qr1 = np.tile(qr,(p_res,1,1))
		integral = np.trapz(qr1 * (Bo-Bc1) * F, dx=dp, axis=1)
		# integral = np.sum(qr1 * (Bo-Bc1) * F, axis=1) * dp

		sigma = np.sum([(Bo-Bm[n]) * am[n] * G[n,:] for n in range(N)], axis=0)

		qt = 1/(2*w*mu*P) * abs(Bc) / (Bo+Bc) * (2*Bo*G[0,:] + integral + sigma)

		# an
		am_prev = am
		am = np.array(
			[ (1/(4*w*mu*P) * np.sum(qt * (Bm[n]-Bc) * G[n,:], axis=0) * dp) for n in range(N) ]
			)
		print 'Delta ao: ', abs(am_prev-am)


		#Qr
		qt1 = np.tile(qt,(p_res,1,1))
		integral = np.trapz(qt1 * (Bc2-Bc1) * F.transpose(1,0,2), dx=dp, axis=1)
		# integral = np.sum(qt1 * (Bc2-Bc1) * F.transpose(), axis=1) * dp
		
		qr = 1/(4*w*mu*P) * (abs(Bc)/Bc) * integral

		'''
		Test Power Conservation
		'''

		error = (1+am[0,:]) * (1-am[0,:].conjugate()) - np.sum(abs(am[1:,:])**2, axis=0) - np.sum((abs(qt)**2 + abs(qr)**2) * np.conjugate(Bc) / abs(Bc), axis=0) * dp
		print 'Power Conservation: ', abs(error.real)

		'''
		Test Gelin, eq. 14
		'''

		print 'Gelin Eq. 14: ', 1/(4*w*mu*P) * np.trapz(qt * (Bm[0]+Bc) * G[0,:], dx=dp, axis=0)

	'''
	Print Stats
	'''

	print '\n----Results Summary----\n'

	print 'Max Power Conservation Error:', np.amax(abs(error.real))

	'''
	Plot results
	'''

	fig, ax = plt.subplots(3,figsize=(7,8))

	ax[0].plot(p/k,qt.real,'r')
	ax[0].plot(p/k,qt.imag,'r:')
	ax[1].plot(p/k,qr.real,'b')
	ax[1].plot(p/k,qr.imag,'b:')

	ax[0].axhline(0,color='k',ls=':')
	ax[1].axhline(0,color='k',ls=':')

	# ax[0].set_xlim(0,10)
	# ax[0].set_ylim(0,1.5)
	# ax[1].set_xlim(0,10)
	# ax[1].set_ylim(0,1.5)

	ax[2].plot(kd,am[0,:],'g')
	ax[2].set_ylim(0,1)

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


if __name__ == '__main__':
  # test_beta()
  # test_beta_marcuse()
  main()
