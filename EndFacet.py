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

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

np.set_printoptions(linewidth=150, precision=8)

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

def beta_marcuse(n,d,wl=1.,pol='TM',Nmodes=None,plot=False):

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

	k = 2*pi/wl
	kappa = np.linspace(0,n*k,10000)
	gamma = lambda x: sqrt((n*k)**2 - x**2 - k**2)
	
	C = n**2 if pol == 'TM' else 1.

	trans = lambda K: tan(K * d) - C * np.real(gamma(K)/K)

	# Find zero crossings, then use the brentq method to find the precise zero crossing 
	# between the two nearest points in the array

	diff = np.diff(np.sign(trans(kappa)))

	Ks = np.array([])

	toggle = True # need to accept alternating zero crossings - the true modes alternate with infinite discontinuites of the tangent function
	for i,idx in enumerate(np.nonzero(diff)[0]):

		if toggle:
			k_low = kappa[idx-1]
			k_high = kappa[idx+1]

			Ks = np.append(Ks, sp.optimize.brentq(trans,k_low,k_high))

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
		print 'Number of modes:', Nmodes

		plt.figure()
		plt.plot(kappa*d/pi, tan(kappa*d))
		plt.plot(kappa*d/pi, sqrt(n**2*k**2 - kappa**2 - k**2)/kappa)
		plt.plot(kappa*d/pi, trans(kappa))
		plt.plot(kappa*d/pi, np.sign(trans(kappa)), 'k:')

		plt.xlabel(r'$\kappa d/\pi$')
		plt.axhline(0, c='k')
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

def numModes(ncore,nclad,kd):
	return np.ceil(sqrt(ncore**2 - nclad**2)*kd/pi).astype(int)

def Reflection(kd,n,incident_mode=0,pol='TE',p_max=20,p_res=1e3,imax=100,convergence_threshold=1e-5,first_order=False, debug=False):
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

	Ns = numModes(n,1,kd)
	N = np.amax(Ns)
	print 'Number of Modes Detected:', N

	# confirm source mode is supported by slab (ie is above cutoff)
	if incident_mode+1 > N:
		print 'Source Mode below cutoff.'
		return np.nan * np.ones(N)
	
	Bo = np.zeros(len(d))
	Bm = np.zeros((np.amax(N),len(d)))

	for j,dj in enumerate(d):
		Bm[:,j] = beta_marcuse(n,dj,wl=wl,Nmodes=N, pol=pol, plot=False)			# Propagation constants of waveguide modes
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

		Am = sqrt(2*w*mu*P / (Bm*d + Bm/gm))

		Bt = sqrt(2*w*mu*P / (pi*abs(Bc))) 
		Br = sqrt(2*p**2*w*mu*P / (pi*abs(Bc)*(p**2*cos(o*d)**2 + o**2*sin(o*d)**2)))
		
		Dr = 1/2. * exp(-1j*p*d) * (cos(o*d) + 1j*o/p * sin(o*d))

	else:

		# Marcuse solution - haven't figured out algebra yet but matches exact solution
		# Am = sqrt(gm/Bm * 2*w*eo*n**2*P / ( n**2 * k**2/(Bm**2 + n**2*gm**2) + gm*d))

		psi = (2*Km*d + sin(2*Km*d))/(4*Km*eps) + cos(Km*d)**2/(2*gm)
		Am = sqrt(w*eo*P/(Bm*psi))

		Bt = sqrt(2*w*eo*P / (pi*abs(Bc))) 
		Br = sqrt(2*p**2*w*eo*P*n**2 / (pi*abs(Bc)*(n**2 * p**2 * cos(o*d)**2 + o**2/n**2 * sin(o*d)**2)))
		
		# Formula with Gelin's "Typo"
		# Br = p * sqrt(2*w*eo*P*eps / (pi*abs(Bc)*(eps**2 * p**2 * cos(o*d)**2 + o**2 * sin(o*d)**2)))

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
		V  = np.zeros((N,len(p),Nds),dtype = 'complex')
		kp = np.zeros((N,len(p),Nds),dtype = 'complex')
	else:
		G = np.zeros((N,len(p),Nds),dtype = 'complex')


	for i in range(N):

		if pol == 'TM':
			# V[i,:,:] = Am[i] * Bt * cos(Km[i]*d) * (gm[i]*cos(p*d)*(gm[i]**2 + Km[i]**2) - p*sin(p*d)*((gm[i]**2 + p**2) + Km[i]**2 - p**2)) \
			# 					 / ((Km[i]**2 - p**2)*(gm[i]**2 + p**2))
			# V[i,:,:] = Am[i] * Bt * cos(Km[i]*d) * (gm[i]*cos(p*d)*(gm[i]**2 + Km[i]**2) - p*sin(p*d)*((gm[i]**2 + p**2)/n**2 + Km[i]**2 - p**2)) \
			# 					 / ((Km[i]**2 - p**2)*(gm[i]**2 + p**2))

			V[i,:,:]  = Am[i] * Bt * cos(Km[i]*d) * ((Km[i]*sin(p*d) - p*cos(p*d)*tan(Km[i]*d))/(eps*(Km[i]**2 - p**2)) \
				           - (p*sin(p*d) - gm[i]*cos(p*d))/(gm[i]**2 + p**2))

			kp[i,:,:] = Am[i] * Bt * cos(Km[i]*d) * ((Km[i]*sin(p*d) - p*cos(p*d)*tan(Km[i]*d))/(     Km[i]**2 - p**2)  \
				           - (p*sin(p*d) - gm[i]*cos(p*d))/(gm[i]**2 + p**2))

		else:
			G[i,:,:] = 2 * k**2 * (eps-1) * Am[i] * Bt * cos(Km[i]*d) * (gm[i]*cos(p*d) - p*sin(p*d)) / ((Km[i]**2 - p**2)*(gm[i]**2 + p**2))


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
	# 								 2*(D1 * (exp(1j*p1*d) * (p2*sin(pd)+1j*p1*cos(pd))) - 1j*p1 ).real / (p1**2-p2**2) )
	
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

		# Gelin's definition of F, with the sin arguments corrected:
		I = np.tile(np.eye(len(p)), (Nds,1,1)).transpose(1,2,0)
		F = 0*I * pi * Bt2 * Br1 * (D1 + Dstar) - \
				np.nan_to_num(k**2 * (eps-1) * Bt2 * Br1 * (sin((o1+p2)*d)/(o1+p2) + sin((o1-p2)*d)/(o1-p2)) / (p1**2-p2**2)) * (1-I)

		

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

			integral = np.trapz(qr1 * (Bo-Bc1) * F, x=p1, axis=1)
			# integral = np.sum(qr1 * (Bo-Bc1) * F, axis=1) * dp

			sigma = np.sum([(Bo-Bm[n]) * am[n] * G[n,:] for n in range(N)], axis=0)

			qt = 1/(2*w*mu*P) * abs(Bc) / (Bo+Bc) * (2*Bo*G[incident_mode,:] + integral + sigma)

			# an
			am_prev = am
			# am = np.array([ (1/(4*w*mu*P) * np.sum(qt * (Bm[n]-Bc) * G[n,:], axis=0) * dp) for n in range(N) ])
			am = np.array([ (1/(4*w*mu*P) * np.trapz(qt * (Bm[n]-Bc) * G[n,:], x=p, axis=0)) for n in range(N) ])

			#Qr
			qt1 = np.tile(qt,(len(p),1,1))
			integral = np.trapz(qt1 * (Bc2-Bc1) * F.transpose(1,0,2), x=p1, axis=1)
			# integral = np.sum(qt1 * (Bc2-Bc1) * F.transpose(1,0,2), axis=1) * dp
			
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
	err2  = 1 - np.sum(abs(am)**2, axis=0) - np.sum(abs(qt**2)*(p<=k) + abs(qr)**2*(p<=k), axis=0) * dp
	print 'Error   in Power Conservation: ', abs(error.real)
	print 'Error 2 in Power Conservation: ', abs(err2.real)
	

	'''
	Test Gelin, eq. 14
	'''

	if TM:
		pass
	else:
		print 'Gelin Eq. 14: ', 1/(4*w*mu*P) * np.trapz(qt * (Bm[incident_mode]+Bc) * G[incident_mode,:], dx=dp, axis=0)

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
		return am
	else:
		return np.nan * np.ones(N)




def main():

	# Define Key Simulation Parameters

	n = sqrt(20)
	
	# kds = np.array([0.837])
	
	# kds = np.array([0.209,0.418,0.628,0.837,1.04,1.25]) # TE Reference values
	kds = np.array([0.314,0.418,0.628,0.837,1.04,1.25]) # TM Reference Values

	# kds = np.linspace(1e-9,2.4,50)
	# kds = np.linspace(0.1,0.5,50)
	# kds = np.linspace(0.5,1.5,50)

	res = 1e3
	incident_mode = 0
	pol='TM'

	imax = 100
	p_max = 100

	ams = putil.stackPoints([
		Reflection(kd,n,
			pol=pol,
			incident_mode=incident_mode,
			p_res=res,
			imax=imax,
			p_max=p_max,
			first_order = True,
			debug=False
			) for kd in kds])

	# fig, ax = plt.subplots(2,figsize=(7,5))
	# [ax[0].plot(kds, abs(am)     , color=colors[i]) for i,am in enumerate(ams)]
	# [ax[1].plot(kds, np.angle(am), color=colors[i]) for i,am in enumerate(ams)]
	# plt.show()
	print ams
	print
	print abs(ams)**2

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


if __name__ == '__main__':
  # test_beta()
  # test_beta_marcuse()
  main()
  # test_f()
