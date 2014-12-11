# !/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib as mpl

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.integrate import quadrature as quad

import putil
import sys
import os
import time

# import pudb; pu.db

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

prop = mpl.font_manager.FontProperties(fname='/Library/Fonts/GillSans.ttc')

np.set_printoptions(linewidth=150, precision=3)

pi = sp.pi
sqrt = sp.emath.sqrt
# mu = 4*pi * 1e-7
# eo = 8.85e-12
# c  = 299792458

mu = 1.
eo = 1.
c  = 1.

# units
cm = 1e-2
um = 1e-6
nm = 1e-9

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

def cot(x):
	return 1/tan(x)

def beta_marcuse(n,d,wl=1.,pol='TM',polarity='even',Nmodes=None,plot=False):

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
	# # Use to force plot/exit when debugging
	# plot=True

	even = (polarity=='even')

	k = 2*pi/wl
	kappa = np.linspace(0,n*k,10000)
	gamma = lambda x: sqrt((n*k)**2 - x**2 - k**2)
	
	C = n**2 if pol == 'TM' else 1.

	if even:
		trans = lambda K: tan(K * d) - C * np.real(gamma(K)/K)
	else:
		trans = lambda K: tan(K * d) + 1/C * np.real(K/gamma(K).real)

	# Find zero crossings, then use the brentq method to find the precise zero crossing 
	# between the two nearest points in the array

	diff = np.diff(np.sign(trans(kappa)))

	Ks = np.array([])

	# need to accept alternating zero crossings - the true modes alternate with infinite discontinuites of the tangent function
	toggle = True
	for i,idx in enumerate(np.nonzero(diff)[0]):

		if toggle and abs(diff[idx])==2:
			k_low = kappa[idx-1]
			k_high = kappa[idx+1]

			try:
				Ks = np.append(Ks, sp.optimize.brentq(trans,k_low,k_high))
			except:
				print 'BrentQ Failed. Plotting Transcendental function...'
				print 'Zero Crossing: (%.3f)' % kappa[idx] * d/pi
				print 'Attempted values of Kd/pi: (%.3f, %.3f)' % (k_low*d/pi, k_high*d/pi)
				print 'Function evaluates to (%.3f, %.3f)' % (trans(k_low),trans(k_high))
				plot=True

		toggle = not toggle

	Bs = sqrt((n*k)**2 - Ks**2)

	# Truncate or pad output as necessary
	if len(Bs) < Nmodes:
		pad = np.zeros(Nmodes - len(Bs)) * np.nan
		Bs = np.hstack((Bs,pad))
		
	elif len(Bs) > Nmodes:
		Bs = Bs[:Nmodes] 

	# Plots for debugging
	if plot:

		plt.ioff()

		print 'Number of modes:', Nmodes
		print Ks*d/pi
		print 'Zero Crossings:', kappa[np.nonzero(diff)[0]] * d/pi

		plt.figure()

		plt.plot(kappa*d/pi, tan(kappa*d))

		if even:
			plt.plot(kappa*d/pi, C * sqrt(n**2*k**2 - kappa**2 - k**2)/kappa)
		else:
			plt.plot(kappa*d/pi, 1/C * (-kappa)/gamma(kappa).real)
			# plt.plot(kappa*d/pi, 1/C * (-kappa)/sqrt(n**2*k**2 - kappa**2 - k**2))

		plt.plot(kappa*d/pi, trans(kappa))
		plt.plot(kappa*d/pi, np.sign(trans(kappa)), 'k:')

		plt.xlabel(r'$\kappa d/\pi$')
		plt.axhline(0, c='k')
		# plt.ylim(-10,10)
		plt.show()
		exit()

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

def numModes(ncore,nclad,kd, polarity='even'):
	# See Marcuse, Light Transmission Optics, p. 310 for graphical origin
	C = 0 if polarity=='even' else pi/2.
	return np.ceil((sqrt(ncore**2 - nclad**2)*kd - C) / pi).astype(int)

def RefQuad(kd,n,incident_mode=0,pol='TE',polarity='even',imax=100,convergence_threshold=1e-5,first_order=False, debug=False):

	# constants
	wl = 10. # Set to 10 to match Gelin values for qt
	k = 2*pi/wl
	d = kd/k
	w = c*k
	eps = n**2

	TM = (pol == 'TM')
	even = (polarity == 'even')
	
	# wavevectors

	Ns = numModes(n,1,kd,polarity)
	N = np.amax(Ns)
	print 'Number of Modes Detected:', N

	# confirm source mode is supported by slab (ie is above cutoff)
	if incident_mode+1 > N:
		print 'Source Mode below cutoff.'
		return np.nan * np.ones(N), [np.nan]
	
	Bm = np.zeros(N)

	Bm = beta_marcuse(n,d,wl=wl,Nmodes=N, pol=pol, polarity=polarity, plot=False)			# Propagation constants of waveguide modes
	Bo = Bm[incident_mode]

	gm  = sqrt(      -k**2 + Bm**2)						# transverse wavevectors in air for guided modes
	Km  = sqrt(n**2 * k**2 - Bm**2)						# transverse wavevectors in high-index region for guided modes

	Bc = lambda p: sqrt(k**2 - p**2)
	o  = lambda p: sqrt((n*k)**2 - Bc(p)**2)

	# Mode Amplitude Coefficients
	P = 1

	if pol == 'TE':
		
		Am = sqrt(2*w*mu*P / (Bm*d + Bm/gm)) # Tested for both even and odd

		if even:
			Bt = lambda p: sqrt(2*w*mu*P / (pi*abs(Bc(p))))
			Br = lambda p: sqrt(2*p**2*w*mu*P / (pi*abs(Bc(p))*(p**2*cos(o(p)*d)**2 + o(p)**2*sin(o(p)*d)**2)))
			Dr = lambda p: 1/2. * exp(1j*p*d) * (cos(o(p)*d) - 1j*o(p)/p * sin(o(p)*d))
		else:
			Bt = lambda p: sqrt(2*w*mu*P * p**2 / (pi*abs(Bc(p))*o(p)**2))
			Br = lambda p: sqrt(2*p**2*w*mu*P / (pi*abs(Bc(p))*(p**2*sin(o(p)*d)**2 + o(p)**2*cos(o(p)*d)**2)))
			Dr = lambda p: 1/2. * exp(1j*p*d) * (sin(o(p)*d) + 1j*o(p)/p * cos(o(p)*d))

	else:
		pass

	# Overlap Integral Solutions, Gm(p) and F(p',p) = F(p1,p2)

	if pol == 'TE':
		if even:
			G = lambda m,p: 2 * k**2 * (eps-1) * Am[m] * Bt(p) * cos(Km[m]*d) * (gm[m]*cos(p*d) - p*sin(p*d)) / ((Km[m]**2 - p**2)*(gm[m]**2 + p**2))

			def _F(P,p):
				if P == p:
					return pi * Bt(p) * Br(P) * 2 * np.real(Dr(P))
				else:
					od = o(P)
					pd = p*d

					return 2*Br(P)*Bt(p) * ((o(P)*sin(od)*cos(pd) - p*cos(od)*sin(pd))/(o(P)**2-p**2) + \
		 								 2*(Dr(p) * (exp(-1j*P*d) * (p*sin(pd)-1j*P*cos(pd))) + 1j*P ).real / (P**2-p**2) )

			F = np.vectorize(_F)

		else:
			pass
	else:
		pass

	'''
	Define recursive functions
	'''

	pmax=100

	# Dictionary of functions
	dict = {}

	if pol == 'TE':

		# QT			

		def qt(p,recursion_level):
			# print 'qt:', recursion_level
			_qr = dict['qr' + str(recursion_level)]
			_am = dict['am' + str(recursion_level)]

			integrand = lambda P,p: qr(P,recursion_level) * (Bo-Bc(P)) * F(P,p)

			return 1/(2*w*mu*P) * abs(Bc(p)) / (Bo+Bc(p)) * (2*Bo*G(incident_mode,p) \
				+ sp.trapz(integrand,0,pmax,args=(p,),points=[p])[0] \
				+ np.sum([(Bo-Bm[n]) * am(recursion_level)[n] * G(n,p) for n in range(N)]))

		# An

		def am(recursion_level):
			# print 'am:', recursion_level
			_qt = dict['qt' + str(recursion_level)]

			if recursion_level == 0: return np.zeros(N, dtype=complex)
			
			integrand = lambda p,m: qt(p,recursion_level-1) * (Bm[m]-Bc(p)) * G(m,p)
		
			return np.array([ 1/(4*w*mu*P) * sp.trapz(integrand,0,pmax,args=(m,))[0] for m in range(N) ])



		# QR

		def qr(p,recursion_level):
			# print 'qr:', recursion_level
			_qt = dict['qt' + str(recursion_level)]

			if recursion_level == 0: return 0

			integrand = lambda P,p: qt(P,recursion_level-1) * (Bc(p)-Bc(P)) * F(p,P)
		
			return 1/(4*w*mu*P) * (abs(Bc(p))/Bc(p)) * sp.trapz(integrand,0,pmax,args=(p,),points=[p])[0]

	else: #TM
		pass

	qt = np.vectorize(qt)
	qr = np.vectorize(qr)

	for i in range(imax):

		if repeat == False:
			converged = True
			break

		print '\nComputing iteration %u of %u' % (i+1,imax)

		dict[qr + str(i)] = qr()




	ps = np.linspace(k,2*k,3)
	qt(ps,0)
	# ans = am(1)
	# print ans
	exit()

	# ps = np.linspace(0,2,100)
	# plt.plot(ps,qt(ps))
	# plt.show()
	# exit()

	# '''
	# Test Convergence
	# '''

	# if first_order:
	# 	converged = True
	# 	break

	# delta = abs(am_prev-am)
	# repeat = True if np.any(delta > convergence_threshold) else False
	# print 'Delta ao:', np.amax(delta)
	
	# # if difference in am has been rising for 2 iterations, value is diverging. Bail.

	# if np.amax(delta) > delta_prev and delta_prev > delta_2prev: break

	# delta_2prev = delta_prev
	# delta_prev = np.amax(delta)

def Reflection(kd,n,incident_mode=0,pol='TE',polarity='even',
	p_max=20,p_res=1e3,imax=100,convergence_threshold=1e-5,first_order=False, debug=False):
	'''

	Solve for amplitudes of all scattered modes resulting from a guided mode incident on the 
	abrupt truncation of a symmetric dielectric slab. Method adapted from Gelin et al (1981): 

	"Rigorous Analysis of the Scattering of Surface Waves in an Abruptly Ended Slab Dielectric Waveguide"

	INPUTS

	kd [np.array] - free-space wavevector x slab half-height
	n  [float]    - slab index
	incident_mode [int] - mode order to use as the source (0 = fundamental; currently treats all modes as even, ie 
		"1" will actually be the mode with 3 field antinodes)

	pol   ['TE' or 'TM']; TM currently not functional.
	p_max [int] - largest transverse wavevector used for infinite integral calculations, in units of k
	p_res [int] - number of p values to be used from 0 to k*p_max
	imax  [int] - maximum number of iterations allowed while testing for convergence
	convergence_threshold [float] - the convergence routine will run until the magnitude of the reflection coefficients drop
		below this value

	first_order [boolean] - if True, don't iterate and simply return the initial solution.

	OUTPUTS

	--note-- will return np.nan upon failure to converge

	am [np.array] - complex reflection coefficients for slab guided modes.

	'''

	'''
	Normalize inputs
	'''
	if type(kd) in [int, float, np.float64]: kd = np.array([kd])
	if type(kd) == list: kd = np.array(kd)

	print kd

	'''
	Calculate wavevectors and other physical quantities needed for all functions
	'''

	# constants
	wl = 10. # Set to 10 to match Gelin values for qt
	k = 2*pi/wl
	d = kd/k
	w = c*k
	eps = n**2

	Nds = len(d)

	TM = (pol == 'TM')

	even = (polarity == 'even')
	
	# # Debugger for mode solver
	# wl = 1.0
	# k = 2*pi/wl
	# kd = 1.0
	# d = kd/k	
	# print beta_marcuse(1.432,d,wl=wl,pol='TE',plot=True) * d
	# print 'Should be 1.21972'

	# print beta_marcuse(1.432,d,wl=wl,pol='TM',plot=True) * d
	# print 'Should be 1.12809'

	# exit()
	

	# wavevectors

	Ns = numModes(n,1,kd,polarity)
	N = np.amax(Ns)
	print 'Number of Modes Detected:', N

	# confirm source mode is supported by slab (ie is above cutoff)
	if incident_mode+1 > N:
		print 'Source Mode below cutoff.'
		return np.nan * np.ones(N), [np.nan]
	
	Bo = np.zeros(len(d))
	Bm = np.zeros((np.amax(N),len(d)))

	for j,dj in enumerate(d):
		Bm[:,j] = beta_marcuse(n,dj,wl=wl,Nmodes=N, pol=pol, polarity=polarity, plot=False)			# Propagation constants of waveguide modes
		Bo[j] = Bm[incident_mode,j]

	gm  = sqrt(      -k**2 + Bm**2)						# transverse wavevectors in air for guided modes
	Km  = sqrt(n**2 * k**2 - Bm**2)						# transverse wavevectors in high-index region for guided modes

	# create an array of wavevectors, p, with two zones of resolution - one below nk where 
	# there are singular features in the overlap integrals, and one above where there are 
	# no high-frequency features

	p_split = np.amax((2*np.amax(Km), 2))

	p = np.hstack((np.linspace(1e-15,p_split*k,p_res),np.linspace(p_split*k + (p_max-p_split)*k/p_res, p_max*k, p_res)))
	p = np.tile(p, (len(d),1))					# transverse wavevectors in air (independent variable for algorithm)
	
	# # Old array creation with single zone.
	# p = np.tile(np.linspace(1e-15,p_max*k,p_res), (len(d),1))					# transverse wavevectors in air (independent variable for algorithm)

	p = p.transpose()

	dp = p[1,0]-p[0,0]
	
	Bc = sqrt(       k**2 - p**2) 						# propagation constants in air for continuum modes
	o  = sqrt(n**2 * k**2 - Bc**2)						# transverse wavevectors in high-index region for continuum modes
	
	# Bc = Bc.real - 1j*Bc.imag
	# o =  o.real  - 1j*o.imag
	# gm = gm.real - 1j*gm.imag
	# Km = Km.real - 1j*Km.imag





	# Mode Amplitude Coefficients
	P = 1

	if pol == 'TE':	

		Am = sqrt(2*w*mu*P / (Bm*d + Bm/gm)) # Tested for both even and odd
		
		if even:
			Bt = sqrt(2*w*mu*P / (pi*abs(Bc)))
			Br = sqrt(2*p**2*w*mu*P / (pi*abs(Bc)*(p**2*cos(o*d)**2 + o**2*sin(o*d)**2)))
			Dr = 1/2. * exp(1j*p*d) * (cos(o*d) - 1j*o/p * sin(o*d))
		else:
			Bt = sqrt(2*w*mu*P * p**2 / (pi*abs(Bc)*o**2))
			Br = sqrt(2*p**2*w*mu*P / (pi*abs(Bc)*(p**2*sin(o*d)**2 + o**2*cos(o*d)**2)))
			Dr = 1/2. * exp(1j*p*d) * (sin(o*d) + 1j*o/p * cos(o*d))
		
		

	else:

		# Marcuse solution - haven't figured out algebra yet but matches exact solution
		# Am = sqrt(gm/Bm * 2*w*eo*n**2*P / ( n**2 * k**2/(Bm**2 + n**2*gm**2) + gm*d))

		psi = (2*Km*d + sin(2*Km*d))/(4*Km*eps) + cos(Km*d)**2/(2*gm)
		Am = sqrt(w*eo*P/(Bm*psi))

		Bt = sqrt(2*w*eo*P / (pi*abs(Bc))) 
		# Br = sqrt(2*p**2*w*eo*P*n**2 / (pi*abs(Bc)*(n**2 * p**2 * cos(o*d)**2 + o**2/n**2 * sin(o*d)**2)))
		
		# Formula with Gelin's "Typo"
		Br = p * sqrt(2*w*eo*P*eps / (pi*abs(Bc)*(eps**2 * p**2 * cos(o*d)**2 + o**2 * sin(o*d)**2)))

		Dr = 1/2. * exp(1j*p*d) * (cos(o*d) - 1j*o/(n**2 * p) * sin(o*d))

	# Plot Amplitudes
	if debug:
		plt.figure()
		plt.plot(p/k,Bt,'b')
		plt.plot(p/k,Br,'r')
		print '\nMode Amplitudes:', Am
		# print sqrt(P*w*eo/(Bm * (1/(eps*4*Km) * (2*Km*d + sin(2*Km*d)) + (cos(Km*d)**2)/(2*gm))))
		# print sqrt(P*w*eo/(Bm * (1/(4*Km) * (2*Km*d + sin(2*Km*d)) + (cos(Km*d)**2)/(2*gm))))


	# Overlap Integral Solutions, Gm(p) and F(p',p) = F(p1,p2)

	if TM:
		if even:
			V  = np.zeros((N,len(p),Nds),dtype = 'complex')
			kp = np.zeros((N,len(p),Nds),dtype = 'complex')
		else:
			pass
	else:
		G = np.zeros((N,len(p),Nds),dtype = 'complex')

	for i in range(N):

		if pol == 'TM':
			V[i,:,:]  = Am[i] * Bt * cos(Km[i]*d) * ((Km[i]*cos(p*d)*tan(Km[i]*d) - p*sin(p*d))/(eps*(Km[i]**2 - p**2)) \
				           - (p*sin(p*d) - gm[i]*cos(p*d))/(gm[i]**2 + p**2))

			kp[i,:,:]     = Am[i] * Bt * cos(Km[i]*d) * ((  Km[i]*cos(p*d)*tan(Km[i]*d) - p*sin(p*d))/(Km[i]**2 - p**2)  \
				            - (p*sin(p*d) - gm[i]*cos(p*d))/(gm[i]**2 + p**2))
  		#G[i,:,:] = 2 * Am[i] * Bt * sin(Km[i]*d) * ((- Km[i]*cot(Km[i]*d)*sin(p*d) + p*cos(p*d))/(Km[i]**2 - p**2)  \
			# 						+ (p*cos(p*d) + gm[i]*sin(p*d))/(gm[i]**2 + p**2))

		else:
			if even:
				G[i,:,:] = 2 * k**2 * (eps-1) * Am[i] * Bt * cos(Km[i]*d) * (gm[i]*cos(p*d) - p*sin(p*d)) / ((Km[i]**2 - p**2)*(gm[i]**2 + p**2))

			else:
				G[i,:,:] = 2 * Am[i] * Bt * sin(Km[i]*d) * ((p*cos(p*d) - Km[i]*cot(Km[i]*d)*sin(p*d))/(Km[i]**2-p**2) \
										+ (gm[i]*sin(p*d) + p*cos(p*d))/(gm[i]**2 + p**2))


	# Convert to multi-dimensional arrays to work with functions of p',p. When transposing, 
	# the index refering to d is kept as the final index

	p1  = np.tile(p, (len(p),1,1)); p2 = p1.transpose(1,0,2)
	Br1 = np.tile(Br,(len(p),1,1))
	Bt1 = np.tile(Bt,(len(p),1,1)); Bt2 = Bt1.transpose(1,0,2)
	o1  = np.tile(o ,(len(p),1,1))
	Bc1 = np.tile(Bc,(len(p),1,1)); Bc2 = Bc1.transpose(1,0,2)
	
	D1  = np.tile(Dr,(len(p),1,1))
	Dstar  = np.conjugate(D1)

	od = o1*d; pd = p2*d.transpose()

	# This method for F is based on a more direct solution of the field overlap integral, with less
	# subsequent algebra:
	# F = 2*Br1*Bt2 * ((o1*sin(od)*cos(pd) - p2*cos(od)*sin(pd))/(o1**2-p2**2) + \
	# 								 2*(D1 * (exp(-1j*p1*d) * (p2*sin(pd)-1j*p1*cos(pd))) + 1j*p1 ).real / (p1**2-p2**2) )
	
	# # # Handle Cauchy singularities
	# cauchy = pi * Bt2 * Br1 * (D1 + Dstar)
	# idx = np.where(1 - np.isfinite(F))
	# F[idx] = 0
	# # F[idx] = cauchy[idx]


	if pol == 'TM':
		f = Br1*Bt2 * ((o1*sin(od)*cos(pd) - p2*cos(od)*sin(pd))/(n**2 * (o1**2-p2**2)) + \
										 2*(D1 * (exp(-1j*p1*d) * (p2*sin(pd)-1j*p1*cos(pd))) + 1j*p1 ).real / (p1**2-p2**2) )

		z = Br1*Bt2 * ((o1*sin(od)*cos(pd) - p2*cos(od)*sin(pd))/        (o1**2-p2**2)  + \
										 2*(D1 * (exp(-1j*p1*d) * (p2*sin(pd)-1j*p1*cos(pd))) + 1j*p1 ).real / (p1**2-p2**2) )


		# Handle Cauchy singularities
		cauchy = pi * Bt2 * Br1 * (D1 + Dstar)
		idx = np.where(1 - np.isfinite(f))
		f[idx] = 0
		# f[idx] = cauchy[idx]

		idx = np.where(1 - np.isfinite(z))
		z[idx] = 0
		# z[idx] = cauchy[idx]


	else:

		if even:

			# # Gelin's definition of F, with the sin arguments corrected:
			# I = np.tile(np.eye(len(p)), (Nds,1,1)).transpose(1,2,0)
			# F = 0*I * pi * Bt2 * Br1 * (D1 + Dstar) - \
			# 		np.nan_to_num(k**2 * (eps-1) * Bt2 * Br1 * (sin((o1+p2)*d)/(o1+p2) + sin((o1-p2)*d)/(o1-p2)) / (p1**2-p2**2)) * (1-I)

			# # Hamid's definition of F, which is apparently actually for TM:
			I = np.tile(np.eye(len(p)), (Nds,1,1)).transpose(1,2,0)
			H   = -k**2 * (eps-1) * Bt2 * Br1 * (sin((o1+p2)*d)/(o1+p2) + sin((o1-p2)*d)/(o1-p2))
			Hpp = H.diagonal(0,0,1)
			Hpp = Hpp.transpose()

			plt.ioff()
			plt.figure()
			plt.imshow(abs(H[:,:,0]-Hpp))
			plt.colorbar()
			plt.figure()
			plt.imshow(np.real(H[:,:,0]-Hpp))
			plt.colorbar()
			plt.show()
			exit()

			# # This method for F is based on a more direct solution of the field overlap integral, with less
			# # subsequent algebra:
			# F = 2*Br1*Bt2 * ((o1*sin(od)*cos(pd) - p2*cos(od)*sin(pd))/(o1**2-p2**2) + \
		 # 								 2*(D1 * (exp(-1j*p1*d) * (p2*sin(pd)-1j*p1*cos(pd))) + 1j*p1 ).real / (p1**2-p2**2) )
		
		else:

			F = 2*Br1*Bt2 * ((p2*sin(od)*cos(pd) - o1*cos(od)*sin(pd))/(o1**2-p2**2) - \
										 2*(D1 * (exp(-1j*p1*d) * (p2*cos(pd)+1j*p1*sin(pd))) -    p1 ).real / (p1**2-p2**2) )
		

		# # # Handle Cauchy singularities
		# cauchy = pi * Bt2 * Br1 * (D1 + Dstar)
		# idx = np.where(1 - np.isfinite(F))
		# # F[idx] = 0
		# F[idx] = cauchy[idx]


	if debug:

		plt.figure()
		# plt.plot(p/k,G[0],'g')
		if TM: 
			plt.plot(p/k,kp[0],'g')
			plt.plot(p/k,V[0],'m')
			plt.plot(p/k,f[:,100,0],'y')
			plt.plot(p/k,z[:,100,0],'c')

			# plt.figure()
			# plt.plot(p/k)

			plt.figure()
			plt.imshow(z[:,:,0].real, extent = putil.getExtent(p/k,p/k), vmin=-1, vmax=1)
			plt.colorbar()

			plt.figure()
			plt.imshow(f[:,:,0].real, extent = putil.getExtent(p/k,p/k), vmin=-1, vmax=1)
			plt.colorbar()

		else:
			plt.figure()
			plt.imshow(F[:,:,0].real, extent = putil.getExtent(p/k,p/k), vmin=-1, vmax=1)
			plt.colorbar()



		plt.show()
		exit()
		







	'''
	Run Neuman series and test for convergence of the fields
	'''

	# Initialize coefficients to zero
	am = np.zeros((N,Nds))
	qr = np.zeros((len(p),Nds))	

	# Remove nan instances for iteration purposes
	nanMask = np.isnan(Bm)

	Bm = np.nan_to_num(Bm)
	Bo = np.nan_to_num(Bo)
	
	if TM:
		V = np.nan_to_num(V)
	else:
		G = np.nan_to_num(G)
		Gm = G[incident_mode]

	if TM:
		vm = V[incident_mode]
		kpm = kp[incident_mode]

	repeat = True; converged = False
	delta_prev = delta_2prev = np.inf

	for i in range(imax):

		if repeat == False:
			converged = True
			break

		print '\nComputing iteration %u of %u' % (i+1,imax)


		if pol == 'TE':

			# Qt

			qr1 = np.tile(qr,(len(p),1,1))

			integrand = qr1 * (Bo-Bc1) * (H-Hpp)/(p1**2-p2**2)
			idx = np.where(1 - np.isfinite(integrand))
			integrand[idx]=0

			integral = np.trapz(integrand, x=p1, axis=1) + qr*(Bo-Bc) * pi*Bt*Br*2*np.real(Dr)

			sigma = np.sum([(Bo-Bm[n]) * am[n] * G[n,:] for n in range(N)], axis=0)

			qt = 1/(2*w*mu*P) * abs(Bc) / (Bo+Bc) * (2*Bo*G[incident_mode,:] + integral + sigma)

			# an
			am_prev = am
			# am = np.array([ (1/(4*w*mu*P) * np.sum(qt * (Bm[n]-Bc) * G[n,:], axis=0) * dp) for n in range(N) ])
			am = np.array([ (1/(4*w*mu*P) * np.trapz(qt * (Bm[n]-Bc) * G[n,:], x=p, axis=0)) for n in range(N) ])

			#Qr
			qt1 = np.tile(qt,(len(p),1,1))

			integrand = qt1 * (Bc2-Bc1) * (H.transpose(1,0,2) - Hpp)/(p2**2 - p1**2)
			idx = np.where(1 - np.isfinite(integrand))
			integrand[idx]=0

			integral = np.trapz(integrand, x=p1, axis=1)
			
			qr = 1/(4*w*mu*P) * (abs(Bc)/Bc) * integral




		else: #TM

			qr1 = np.tile(qr,(len(p),1,1))
			integral = np.trapz(qr1 * (Bo*vm*z - Bc1*kpm*f), x=p1, axis=1)

			sigma = np.sum([(Bo*vm*kp[n] - Bm[n]*V[n]*kpm) * am[n] for n in range(N)], axis=0)

			qt = 1/(w*eo*P) * abs(Bc) / (Bo*vm+Bc*kpm) * (2*Bo*vm*kpm + integral + sigma)

			# an
			am_prev = am
			am = np.array([ (1/(2*w*eo*P) * np.trapz(qt * (Bm[n]*V[n] - Bc*kp[n]), x=p, axis=0)) for n in range(N) ])

			#Qr
			qt1 = np.tile(qt,(len(p),1,1))
			integral = np.trapz(qt1 * (Bc2 * f.transpose(1,0,2) - Bc1 * z.transpose(1,0,2)), x=p1, axis=1)
			
			qr = 1/(2*w*eo*P) * (abs(Bc)/Bc) * integral


		'''
		Test Convergence
		'''

		if first_order:
			converged = True
			break

		delta = abs(am_prev-am)
		repeat = True if np.any(delta > convergence_threshold) else False
		print 'Delta ao:', np.amax(delta)
		
		# if difference in am has been rising for 2 iterations, value is diverging. Bail.

		if np.amax(delta) > delta_prev and delta_prev > delta_2prev: break

		delta_2prev = delta_prev
		delta_prev = np.amax(delta)



	'''
	Test Power Conservation
	'''

	error = (1+am[0,:]) * (1-am[0,:].conjugate()) - np.sum(abs(am[1:,:])**2, axis=0) - np.sum((abs(qt)**2 + abs(qr)**2) * np.conjugate(Bc) / abs(Bc), axis=0) * dp
	print 'Error   in Power Conservation: ', abs(error.real)
	

	'''
	Test Gelin, eq. 14
	'''

	if TM:
		# equation A14 in Gelin's paper seems to be off by 1/2, thus preceeding factor of 2
		shouldBeOne = 2 * 1/(4*w*eo*P) * np.trapz(qt*(Bc*kpm + Bo*vm), x=p, axis=0)
	else:
		shouldBeOne = 1/(4*w*mu*P) * np.trapz(qt * (Bo+Bc) * Gm, x=p, axis=0)

	print 'Gelin Eq. 14: ', shouldBeOne

	'''
	Print Stats
	'''

	print '\n----Results Summary----\n'

	print 'Max Power Conservation Error:', np.amax(abs(error.real))

	'''
	Plot results
	'''

	# Remove values for cutoff modes
	am[np.nonzero(nanMask)] = complex(np.nan, np.nan)

	# scattering coefficients

	if debug:
		print '\n Guided Mode Coefficients:'
		fig, ax = plt.subplots(2,figsize=(7,5))

		ax[0].plot(p/k,qt.real,'r')
		ax[0].plot(p/k,qt.imag,'r:')
		ax[1].plot(p/k,qr.real,'b')
		ax[1].plot(p/k,qr.imag,'b:')

		ax[0].axhline(0,color='k',ls=':')
		ax[1].axhline(0,color='k',ls=':')

		print am

	# ax[0].set_xlim(0,10)
	# ax[0].set_ylim(0,1.5)
	# ax[1].set_xlim(0,10)
	# ax[1].set_ylim(0,1.5)

	# Reflection Magnitude and Phase

	# refFig, refAx = plt.subplots(2,figsize=(7,5))

	# for j in range(N):
	# 	refAx[0].plot(kd,np.abs(  am[j,:]),color=colors[j],marker='o')
	# 	refAx[1].plot(kd,np.angle(am[j,:]) / pi,color=colors[j],marker='o')
	
	# refAx[0].set_ylim(0,1)

	# plt.show()

	if converged:
		return am, shouldBeOne[0]
	else:
		return [np.nan * np.ones(N)], [np.nan]

def main():

	# Define Key Simulation Parameters
	
	# kds = np.array([3.])
	
	# kds = np.array([0.209,0.418,0.628,0.837,1.04,1.25]) # TE Reference values
	# kds = np.array([0.314,0.418,0.628,0.837,1.04,1.25]) # TM Reference Values

	# To convert into the d = h/wl notation for dielectricSlab, use:
	# kd = d*pi/n

	kds = np.linspace(0.72,2.4,100)
	# kds = np.linspace(0.1,0.5,50)
	# kds = np.linspace(1e-15,3,50)

	n = sqrt(20)

	res = 300
	incident_mode = 0
	pol='TE'
	polarity = 'even'

	imax = 200
	p_max = 10

	plt.ion()
	

	fig, ax = plt.subplots(3,figsize=(4.5,8))
	ax[2].axhline(1, color='k', ls = ':')
	plt.show()

	ams = []
	accuracy = []

	method = ReflectionWithHamidsCorrections

	for kdi,kd in enumerate(kds):

		print '\nkd:', kd

		a, acc = method(kd,n,
										pol=pol,
										polarity=polarity,
										incident_mode=incident_mode,
										p_res=res,
										imax=imax,
										p_max=p_max,
										first_order=False,
										debug=False
										)

		ams.append(a)
		accuracy.append(np.abs(acc))

		data = putil.stackPoints(ams)
		data = np.array(data)

		[ax[0].plot(kds[:kdi+1], abs(data[i])        , color=colors[i], marker='o') for i,am in enumerate(data)]
		[ax[1].plot(kds[:kdi+1], np.angle(data[i])/pi, color=colors[i], marker='o') for i,am in enumerate(data)]
		ax[0].set_ylim(0,1.2)
		
		[ax[2].plot(kds[:kdi+1], accuracy, color='b', marker='o', lw=2) for i,am in enumerate(data)]
		ax[2].set_ylim(0,1.5)

		fig.canvas.draw()

	plt.ioff()
	plt.show()

	## Output for Comparison with Gelin's Table
	# print ams
	# print
	# print abs(ams)**2

	## export data
	output = np.vstack((kds,data,np.array(accuracy, dtype=complex)))
	print output

	sname = putil.DATA_PATH + '/Rectangular Resonator/End Facet Reflection Coefs/endFacet_a' + str(incident_mode) + '_' + pol + '_' + polarity + '_' + str(n) + '.txt'
	np.savetxt(sname,output)


'''
TODO

plot qt*dp instead of qt when comparing to Gelin figure
bug when submitting multiple kds - returns nan
Fix Gelin error equation for odd modes
mode solver error for odd TE when second mode appears
use mag ao for convergence?
compare RCWA
output to file
optimization of resolution - pick one kd and sweep
fix error for int type kds
'''

def PrettyPlots():

	n = sqrt(20)
	
	incident_mode = 0
	pol='TE'
	polarity = 'even'


	fname = putil.DATA_PATH + '/Rectangular Resonator/End Facet Reflection Coefs/endFacet_a' + str(incident_mode) + '_' + pol + '_' + polarity + '_' + str(n) + '.txt'

	with open(fname,'r') as f:
		s = f.read()

	s = s.replace('+-','-')
	s = s.replace('(','')
	s = s.replace(')','')

	with open(putil.DATA_PATH + '/endfacet_temp.txt','w') as f:	
		f.write(s)


	data = np.loadtxt(putil.DATA_PATH + '/endfacet_temp.txt', dtype=complex)
	
	N = data.shape[0] - 2

	kd = data[0,:]
	err = data[-1,:]

	idx = np.arange(N) + 1

	fig, ax = plt.subplots(2,sharex=True,figsize=(4.5,7))

	for i in idx:
		ax[0].plot(kd,  abs(data[i,:])        , color=colors[i-1], lw=2)
		ax[1].plot(kd, -np.angle(data[i,:])/pi, color=colors[i-1], lw=2)

	ax[1].axhline(0, color='k', ls=':')

	# ax[0].set_xlim(0.6,1.5)
	ax[0].set_ylim(0,1.1)
	
	# ax[1].set_ylim(1/6.,2/3.)

	ax[1].set_xlabel(r'$k_{o}d$')

	ax[0].set_ylabel(r'$|a_{n}|$', fontsize=14)
	ax[1].set_ylabel(r'$\angle a_{n}$', fontsize=14)

	labels = [r'$a_{%u}$' % (2*j) for j in range(3)]
	ax[1].legend(labels, loc='lower left')

	ax[0].set_title(pol + ' - ' + r'Source Mode: $a_{%u}$' % 2*incident_mode, fontproperties=prop, fontsize=18)

	plt.tight_layout()
	plt.show()

def test_v():

	kd = 1.

	n = sqrt(20)

	wl = 10. # Set to 10 to match Gelin values for qt
	k = 2*pi/wl
	d = kd/k
	w = c*k
	eps = n**2

	k = 2*pi/wl
	d = 1.

	P = 1

	Bm = beta_marcuse(n,d,wl=wl, Nmodes=1, pol='TM', plot=False)			# Propagation constants of waveguide Modes

	p_max = 20
	p_res = 300
	ps = np.linspace(1e-15,p_max*k,p_res)

	I = np.zeros(p_res)

	x = np.linspace(0,10*d,10000)
	dx = x[1]-x[0]


	for i,p in enumerate(ps):
		# print i

		Bc = sqrt(       k**2 - p**2) 						# propagation constants in air for continuum modes
		o  = sqrt(n**2 * k**2 - Bc**2)						# transverse wavevectors in high-index region for continuum modes
		
		gm  = sqrt(      -k**2 + Bm**2)						# transverse wavevectors in air for guided modes
		Km  = sqrt(n**2 * k**2 - Bm**2)						# transverse wavevectors in high-index region for guided modes

		A = sqrt(gm/Bm * 2*P*w*eo*eps / ((n*k)**2/(Bm**2 + (n*gm)**2) + gm*d))

		Bt = sqrt(2*w*eo*P/(pi*abs(Bc)))

		Br = p * sqrt(2*w*eo*eps*P / (pi * Bm * (eps * p**2 * cos(o*d)**2 + o**2/eps * sin(o*d)**2)))

		D = 1/2. * (cos(o*d) + 1j * o/(p*n**2)*sin(o*d)) * exp(-1j*p*d)


		Hm = A * cos(Km*x) * (x < d) + A * exp(gm*d) * cos(Km*d) * exp(-gm*x) * (x >= d)

		Hr = Br * cos(o*x) * (x < d) + 2 * (Br * D * exp(1j*p*x)).real * (x >= d)

		Ht = Bt * cos(p*x)



		# perform overlap

		integrand = (Hm * Ht.conj())/n**2 * (x <= d) + (Hm * Ht.conj()) * (x >= d)
		I[i] = np.trapz(integrand, dx=dx)

		# plt.plot(Hm.real,x/d,'r')
		# plt.plot(Hp.real,x/d,'b')
		# plt.plot(integrand * A/np.amax(integrand),x/d,'g')
		# plt.axhline(1,c='k',ls=':')
		# plt.axvline(0,c='k',ls=':')
		# plt.show()

	plt.plot(ps/k,I,'ro')
	plt.show()

def test_f():

	kd = 1.

	n = sqrt(20)

	wl = 10. # Set to 10 to match Gelin values for qt
	k = 2*pi/wl
	d = kd/k
	w = c*k
	eps = n**2

	k = 2*pi/wl
	d = 1.

	P = 1

	Bm = beta_marcuse(n,d,wl=wl, Nmodes=1, pol='TM', plot=False)			# Propagation constants of waveguide Modes

	p_max = 4
	p_res = 600
	ps = np.linspace(1e-15,p_max*k,p_res)

	p2 = 0.62863285

	xres = d/100.
	xmax = np.linspace(300,302,2)

	I = np.zeros((p_res,len(xmax)),dtype=complex)

	for j,xm in enumerate(xmax):
		x = np.arange(0,xm*d,xres)
		dx = x[1]-x[0]

		print j

		for i,p in enumerate(ps):
			# print i

			Bc = sqrt(       k**2 - p**2) 						# propagation constants in air for continuum modes
			o  = sqrt(n**2 * k**2 - Bc**2)						# transverse wavevectors in high-index region for continuum modes
			
			Bc2 = sqrt(       k**2 - p2**2) 						# propagation constants in air for continuum modes
			o2  = sqrt(n**2 * k**2 - Bc2**2)						# transverse wavevectors in high-index region for continuum modes
			
			gm  = sqrt(      -k**2 + Bm**2)						# transverse wavevectors in air for guided modes
			Km  = sqrt(n**2 * k**2 - Bm**2)						# transverse wavevectors in high-index region for guided modes

			A = sqrt(gm/Bm * 2*P*w*eo*eps / ((n*k)**2/(Bm**2 + (n*gm)**2) + gm*d))

			Bt = sqrt(2*w*eo*P/(pi*abs(Bc2)))

			Br = p * sqrt(2*w*eo*eps*P / (pi * Bm * (eps * p**2 * cos(o*d)**2 + o**2/eps * sin(o*d)**2)))

			D = 1/2. * (cos(o*d) + 1j * o/(p*n**2)*sin(o*d)) * exp(-1j*p*d)


			Hr = Br * cos(o*x) * (x < d) + 2 * (Br * D * exp(1j*p*x)).real * (x >= d)

			Ht = Bt * cos(p2*x)



			# perform overlap

			integrand = (Hr * Ht.conj())/n**2 * (x <= d) + (Hr * Ht.conj()) * (x >= d)
			I[i,j] = np.trapz(integrand, dx=dx)

	# plt.plot(Ht.real,x/d,'r')
	# plt.plot(Hr.real,x/d,'b')
	# plt.plot(integrand.real * Bt/np.amax(integrand),x/d,'g')
	# plt.axhline(1,c='k',ls=':')
	# plt.axvline(0,c='k',ls=':')

	# plt.figure()
	# plt.imshow(I.real,aspect='auto',interpolation='nearest')

	plt.plot(ps/k,I.real)
	plt.show()

def test_draw():

	import time

	# mpl.use('MacOSX')
	plt.ion()
	plt.figure()
	plt.show()

	a = np.array([1,2,3])
	plt.plot(a,'ro')
	plt.draw()

	time.sleep(3)

	# plt.plot(2*a,'bo')
	# plt.draw()

	# time.sleep(1)

	plt.ioff()
	plt.show()

def test_quad():

	kd = 0.628

	n = sqrt(20)

	incident_mode = 0
	pol='TE'
	polarity = 'even'

	imax = 100

	RefQuad(kd,n,incident_mode,pol,polarity)

def convergence_test_single():
	'''
	For a specific value of kd, sweep pmax and pres and plot the result
	'''

	kd = 0.15

	n = sqrt(20)

	res = np.arange(1,15,1) * 1e2
	p_max = np.arange(5,31,2.5)

	incident_mode = 0
	pol='TE'
	polarity = 'even'

	imax = 200

	plt.ion()

	fig, ax = plt.subplots(3,figsize=(10,8),sharex=True,sharey=True)
	plt.show()

	aos = np.zeros((len(res),len(p_max)),dtype=complex) * complex(np.nan,np.nan)
	accuracy = np.zeros((len(res),len(p_max)),dtype=complex) * complex(np.nan,np.nan)

	mag = ax[0].imshow(abs(aos), aspect='auto')
	divmag = make_axes_locatable(ax[0])
	cmag = divmag.append_axes("right", size="1%", pad=0.05)
	cbarmag = plt.colorbar(mag,cax=cmag,format='%.2f')

	phase = ax[1].imshow(np.angle(aos)/pi, aspect='auto')
	divphase = make_axes_locatable(ax[1])
	cphase = divphase.append_axes("right", size="1%", pad=0.05)
	cbarphase = plt.colorbar(phase,cax=cphase,format='%.2f')

	error = ax[2].imshow(abs(accuracy), aspect='auto')
	diverror = make_axes_locatable(ax[2])
	cerror = diverror.append_axes("right", size="1%", pad=0.05)
	cbarerror = plt.colorbar(error,cax=cerror,format='%.2f')

	ax[2].set_xlabel(r'Max$\{\rho\}$')
	ax[1].set_ylabel('res')


	for i,res_i in enumerate(res):
		for j,pmax_j in enumerate(p_max):

			a, acc = Reflection(kd,n,
													pol=pol,
													polarity=polarity,
													incident_mode=incident_mode,
													p_res=res_i,
													imax=imax,
													p_max=pmax_j,
													first_order=False,
													debug=False
													)

			try:
				aos[i,j] = a[0][0]
				accuracy[i,j] = acc
			except TypeError:
				pass

			# |ao|
			mag.set_data(abs(aos))
			mag.set_extent(putil.getExtent(p_max,res))
			mag.autoscale()

			# phase ao
			phase.set_data(np.angle(aos)/pi)
			phase.set_extent(putil.getExtent(p_max,res))
			phase.autoscale()

			# error
			error.set_data(abs(accuracy))
			error.set_extent(putil.getExtent(p_max,res))
			error.autoscale()

			fig.canvas.draw()

	plt.ioff()
	plt.show()

def smoothMatrix(M):
	'''
	Helper function for Reflection. Takes integrand matrix, M, and for each point on the main
	diagonal sets the average of the two adjacent indices on axis 0.

	M must be square
	'''
	N = len(M[:,0])

	for n in range(N):
		if n==0: M[n,n] = M[n+1,n]
		elif n > 0 and n < N-1: M[n,n] = (M[n+1,n] + M[n-1,n]) / 2.
		elif n==N-1: M[n,n] = M[n-1,n]
	
	return M

def ReflectionWithHamidsCorrections(kd,n,incident_mode=0,pol='TE',polarity='even',
	p_max=20,p_res=1e3,imax=100,convergence_threshold=1e-5,first_order=False, debug=False):

	'''
	Major code refactor for cleanliness. TE even modes only.
	'''

	debug = False

	m = incident_mode

	# constants
	wl = 10. # Set to 10 to match Gelin values for qt
	k = 2*pi/wl
	d = kd/k
	w = c*k
	eps = n**2

	# Solve slab for all modes
	N = numModes(n,1,kd)
	B = beta_marcuse(n,d,wl=wl,pol='TE',polarity='even',Nmodes=N,plot=False)

	# Define Wavevectors
	g  = sqrt(      -k**2 + B**2)						# transverse wavevectors in air for guided modes
	K  = sqrt(n**2 * k**2 - B**2)						# transverse wavevectors in high-index region for guided modes

	Bc = lambda p: sqrt(k**2 - p**2)
	o  = lambda p: sqrt((n*k)**2 - Bc(p)**2)

	# Mode Amplitude Coefficients
	P = 1

	A = sqrt(2*w*mu*P / (B*d + B/g)) # Tested for both even and odd

	Bt = lambda p: sqrt(2*w*mu*P / (pi*abs(Bc(p))))
	Br = lambda p: sqrt(2*p**2*w*mu*P / (pi*abs(Bc(p))*(p**2*cos(o(p)*d)**2 + o(p)**2*sin(o(p)*d)**2)))
	Dr = lambda p: 1/2. * exp(-1j*p*d) * (cos(o(p)*d) + 1j*o(p)/p * sin(o(p)*d))

	# Define Helper functions for integrals

	G = lambda m,p: 2 * k**2 * (eps-1) * A[m] * Bt(p) * cos(K[m]*d) * (g[m]*cos(p*d) - p*sin(p*d)) / ((K[m]**2 - p**2)*(g[m]**2 + p**2))

	def Ht(qr,a,b):

		'''
		qr is taken in as simply qr(p). To correspond to qr(p') in this 2D matrix formulation, it needs
		to be reshaped into a 2D matrix, whose values are dependent by row (0th axis), but not by column.
		'''
		qr = np.tile(qr,(pres,1)).transpose()

		return -qr * (B[m] - Bc(a)) * k**2 * (eps-1) * Bt(b) * Br(a) * (  sin( (o(a)+b) *d)/(o(a)+b)  +  sin( (o(a)-b) *d)/(o(a)-b)  )

	def Hr(qt,a,b):

		qt = np.tile(qt,(pres,1)).transpose()

		return -qt * (Bc(b) - Bc(a)) * k**2 * (eps-1) * Bt(b) * Br(a) * (  sin( (o(a)+b) *d)/(o(a)+b)  +  sin( (o(a)-b) *d)/(o(a)-b)  )


	# Define mesh of p values
	pmax = p_max*k
	pres = p_res
	p = np.linspace(1e-3,pmax,pres)

	'''
	2D mesh of p values for performing integration with matrices. Rows correspond to
	the value of p', columns to p (for H(p',p)).

	Ex. H(p',p)[3,2] chooses the value of H with the 4th value of p' and the 3rd value of p.

	Integrating over p' would then amount to summing over axis=0. This sums the rows of the matrix H, producing 
	a row vector representing a function of p.
	'''
	p2,p1 = np.meshgrid(p,p) 

	# Test Helper Function evaluation on array objects
	# Works without vectorization for single kd input
	if debug:

		print 'Test axes of matrices - p1 should change with row (1st index):'
		print 'p1[0,0]:', p1[0,0]
		print 'p1[1,0]:', p1[1,0]
		print 'p1[0,1]:', p1[0,1]

		print '\nG'
		print G(0,p)
		print '\np1'
		print p1
		print '\np2'
		print p2
		print '\nH'
		print H(1,p1,p2)
		print '\nHpp'
		print H(1,p2,p2)

	# Define initial states for an, qr
	a = np.zeros(N, dtype=complex)
	qr = np.zeros(pres, dtype=complex)

	'''
	Iteratively define qt,a,qr until the value of a converges to within some threshold.
	'''
	repeat = True
	converged = False
	delta_prev = delta_2prev = np.inf

	for i in range(imax):
		print '\nComputing iteration %u of %u' % (i+1,imax)

		if not repeat:
			break

		# TODO - process the lead terms before looping, since they are independent of the iteration
		
		integrand = (Ht(qr,p1,p2) - Ht(qr,p2,p2))/(p1**2 - p2**2) # blows up at p1=p2
		integrand = smoothMatrix(integrand)											# elminate singular points by using average of nearest neighbors.

		if debug:
			print integrand
			print np.trapz(integrand,dx=1, axis=0)
		
		# if i == 1:
		# 	plt.ioff()
		# 	plt.figure()

		# 	plt.plot(p/k,abs(integrand[0,:]),'r')
		# 	plt.plot(p/k,abs(integrand[:,0]),'b')

		# 	plt.figure()
		# 	# print sp.log10(abs(integrand[250,0]))
		# 	# print sp.log10(abs(integrand[250,1]))
		# 	# plt.imshow(sp.log10(abs(Ht(qr,p1,p2))), extent = putil.getExtent(p/k,p/k))
		# 	plt.imshow(sp.log10(abs(integrand)), extent = putil.getExtent(p/k,p/k))
		# 	plt.colorbar()
		# 	plt.show()
		# 	exit()

		qt = 1/(2*w*mu*P) * abs(Bc(p))/(B[m]+Bc(p)) * ( \
			2*B[m]*G(m,p) \
			+ np.sum([  (B[m]-B[j])*a[j]*G(j,p) for j in range(N) ]) \
			# + sp.integrate.quad(integrand, x=p, axis=0) \
			+ np.trapz(integrand, x=p, axis=0) \
			+ qr * (B[m]-Bc(p)) * pi * Bt(p)*Br(p)*2*np.real(Dr(p))
			)

		a_prev = a
		a = [1/(4*w*mu*P) * np.trapz(qt * (B[j]-Bc(p)) * G(j,p), x=p) for j in range(N)]; a = np.array(a);

		integrand = (Hr(qt,p2,p1) - Hr(qt,p2,p2))/(p2**2 - p1**2) # blows up at p1=p2
		integrand = smoothMatrix(integrand)

		if i == 2:
			plt.ioff()
			plt.figure()

			plt.plot(p/k,abs(integrand[0,:]),'r')
			plt.plot(p/k,abs(integrand[:,0]),'b')
			plt.plot(p/k, Br(p),'g')

			plt.figure()
			# print sp.log10(abs(integrand[250,0]))
			# print sp.log10(abs(integrand[250,1]))
			# plt.imshow(sp.log10(abs(Ht(qr,p1,p2))), extent = putil.getExtent(p/k,p/k))
			plt.imshow(sp.log10(abs(integrand)), extent = putil.getExtent(p/k,p/k))
			plt.colorbar()
			plt.show()
			exit()

		qr = 1/(4*w*mu*P) * abs(Bc(p))/Bc(p) * np.trapz(integrand, x=p, axis=0)

		# Test for convergence
		delta = abs(a_prev-a)
		print 'Delta a:', np.amax(delta)
		if not np.any(delta > convergence_threshold):
		 repeat = False
		 converged = True			

		# if difference in am has been rising for 2 iterations, value is diverging. Bail.

		if np.amax(delta) > delta_prev and delta_prev > delta_2prev: break

		delta_2prev = delta_prev
		delta_prev = np.amax(delta)


	'''
	Loop completed. Perform Error tests
	'''

	shouldBeOne = 1/(4*w*mu*P) * np.trapz(qt*(B[m]+Bc(p))*G(m,p), x=p)

	'''
	Output results
	'''

	print "\n--- Outputs ---"
	if debug:
		print '|a|:', abs(a)
		print '<a:', np.angle(a)
		print 'shouldBeOne:', shouldBeOne

	if converged:
		return a, np.array(shouldBeOne)
	else:
		return a * np.nan, np.array(np.nan)



if __name__ == '__main__':
	# ReflectionWithHamidsCorrections(0.0242424252323,sqrt(20),p_res=500,p_max=5)
  # test_quad()
  # test_beta_marcuse()
  # convergence_test_single()
  main()
  # PrettyPlots()
	# dummy()