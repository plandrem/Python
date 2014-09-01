#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os
import pickle
import DipoleInterference as DI

from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'


# Polarization according to waveguide convention - TE = E out of screen (except when loading FDFD fields)
# Theta is the angle between the slab surface (z axis) and the guided ray (ie, the compliment of the angle of incidence)
# Note that this is the opposite of the resonator/FDFD naming convention

# All wavevectors are normalized (divided) by ko (ie k_cladding)
# All thicknesses are normalized by high-index wavelength (ie d is actually d*n/lambda_o)

'''
Load Reflection Coefficient data
0th Index - Polarization [TE, TM]
1st Index - Refractive Index [2,4,8]
2nd Index - Mode Order [0,1,2,3]
3rd Index - data, value at height/high-index wl (d)

ds values are defined as linspace(0.02,10,101)
'''
POLS 	= np.array(['TE','TM'])
NCORES 	= np.array([2,4,8])
ORDERS 	= np.array(range(5))
NSUBS   = np.array([1,2,10])	#note - 10 represents 10j

PATH = '/Users/Patrick/Documents/PhD/Data/Rectangular Resonator/'

R_ds = np.loadtxt(PATH + 'reflection_coefficient_ds.txt')
#R_ds = np.linspace(0.02,10,101)
R_data = np.zeros((2,len(NCORES),len(ORDERS),len(NSUBS),len(R_ds)),dtype=complex)

PATH = '/Users/Patrick/Documents/PhD/Data/Rectangular Resonator/'
# PATH = 'C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\'

for poli,pol in enumerate(POLS):
	for nci,nc in enumerate(NCORES):
		for oi,o in enumerate(ORDERS):
			for nsi,ns in enumerate(NSUBS):
				fn = PATH + 'reflection_coefficient_' + pol + '_n_%u_mode_%u_nsub_%u.txt' % (nc,o,ns)
				
				if os.path.exists(fn): R_data[poli,nci,oi,nsi,:] = np.loadtxt(fn,delimiter=',',dtype=complex)

'''
R_n4_TM = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_TM_extended.csv',delimiter=',',dtype=complex)
R_n4_TM1 = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_TM_mode_2.txt',delimiter=',',dtype=complex)
R_n4_TM2 = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_TM_mode_3.txt',delimiter=',',dtype=complex)
R_n4_TM3 = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_TM_mode_4.txt',delimiter=',',dtype=complex)

R_n4_ds = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_ds_extended.txt')
R_n4_ds1 = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_ds_mode_2.txt')
R_n4_ds2 = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_ds_mode_3.txt')
R_n4_ds3 = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_ds_mode_4.txt')
'''
def numModes(ncore,nclad,d):
	# Caution - have noticed inconsistencies with D. Marcuse calculation of number of modes! - see EndFacet.py
	critical_angle = sp.arcsin(nclad/ncore)
	M = sp.sin(pi/2. - critical_angle) / (1/(2*d))
		
	return np.ceil(M).astype(int)
	
def numTotalModes(ncore,nclad,d):
	
	# TODO - needs to be modified for asymmetric case
	M = 2*d
		
	return int(np.floor(M))+1
	
		
def phase(d,pol='TM',mode=0,n=4,nsub=1):
	'''
	returns reflection phase in degrees as a function of waveguide thickness (units of high index wavelength)
	
	Computed using Junghyun Park's Matlab RCWA code
	'''
	
	try:
		poli 	= np.where(POLS==pol)[0][0]
		mi 		= np.where(ORDERS==mode)[0][0]
		ni 		= np.where(NCORES==n)[0][0]
		nsi		= np.where(NSUBS==nsub)[0][0]
	except IndexError:
		print "Phase: Selected data does not exist (Polarization: " + pol + ", mode: %u, n: %u)" % (mode,n)
		return 0
	
	angle = np.angle(R_data[poli,ni,mi,nsi,:],deg=True)
	interp = interp1d(R_ds,angle,kind='cubic')

	p = interp(d)

	nclad = 1
	p = guidedOnly(p,n,nclad,d,mode,pol=pol,nsub=nsub)
	
	return p
	
def example_phase():

	plt.figure(figsize=(6,5))
	
	xs = np.linspace(0.02,3,1000)
	
	#for ni,n in enumerate([2,4,8]):
	for mi,m in enumerate([0,1,2,3]):
		ys = phase(xs,pol='TM',mode=mi,n=4)/180

		plt.plot(xs,ys,color=colors[mi],lw=2)

	plt.ylim(0,0.7)
	plt.title('Reflection Phase')
	plt.xlabel(r'Height/$\lambda$')
	plt.ylabel(r'Angle(R) (Rad/$\pi$)')
	plt.legend((r'$n_{core}$ = 2',r'$n_{core}$ = 4',r'$n_{core}$ = 8'))
	plt.show()

def example_phase_vs_mode():

	plt.figure(figsize=(6,5))
	
	xs = np.linspace(0.02,3,1000)
	
	for mi,m in enumerate([0,1,2,3]):
		ys = phase(xs,pol='TM',mode=mi,n=4)/180

		plt.plot(xs,ys,color=colors[mi],lw=2)

	plt.ylim(0,1)
	plt.title('Reflection Phase')
	plt.xlabel(r'Height/$\lambda$')
	plt.ylabel(r'Angle(R) (Rad/$\pi$)')

	leg = ['m = ' + str(x) for x in range(4)]
	plt.legend(leg,loc='best')
	plt.show()

def effect_of_mirrors_on_phase():

	plt.figure(figsize=(6,5))
	
	xs = np.linspace(0.02,3,1000)
	
	y1 = phase(xs,pol='TM',mode=0,n=4,nsub=1)/180
	y2 = phase(xs,pol='TM',mode=0,n=4,nsub=2)/180
	#TODO - add metal

	plt.plot(xs,y1,color='r',lw=2)
	plt.plot(xs,y2,color='b',lw=2)

	plt.ylim(0,1)
	plt.title('Reflection Phase')
	plt.xlabel(r'Height/$\lambda$')
	plt.ylabel(r'Angle(R) (Rad/$\pi$)')
	plt.legend((r'$n_{sub}$ = 1',r'$n_{sub}$ = 2'))
	plt.show()
	
def effect_of_mirrors_on_R():

	plt.figure(figsize=(6,5))
	
	xs = np.linspace(0.02,3,1000)
	
	y1 = Ref(xs,pol='TM',mode=0,n=4,nsub=1)
	y2 = Ref(xs,pol='TM',mode=0,n=4,nsub=2)
	#TODO - add metal

	plt.plot(xs,y1,color='r',lw=2)
	plt.plot(xs,y2,color='b',lw=2)

	plt.ylim(0,1)
	plt.title('Reflection')
	plt.xlabel(r'Height/$\lambda$')
	plt.ylabel('|R|')
	plt.legend((r'$n_{sub}$ = 1',r'$n_{sub}$ = 2'))
	plt.show()

def guidedOnly(data,n,nclad,d,mode,pol,nsub):
	'''
	Take some input array of data corresponding to different thicknesses and eliminate
	any data associated with modes below cutoff

	Note about mode order -- Beta assumes 1 as the fundamental, while the RCWA methods use 0.  This
	function primarily interfaces with RCWA, and thus uses the 1-indexed convention
	'''

	
	bs = getBetas(n,nclad,d,[mode+1],pol=pol,nsub=nsub)[:,0,0]

	mask = np.isnan(bs)

	data[np.nonzero(mask)]=np.nan

	return data



def Ref(d,pol='TM',mode=0,n=4,nsub=1):
	'''
	returns reflection magnitude as a function of waveguide thickness (units of high index wavelength)
	
	Computed using Junghyun Park's Matlab RCWA code
	'''
	
	try:
		poli 	= np.where(POLS==pol)[0][0]
		mi 		= np.where(ORDERS==mode)[0][0]
		ni 		= np.where(NCORES==n)[0][0]
		nsi		= np.where(NSUBS==nsub)[0][0]
	except IndexError:
		print "Ref: Selected data does not exist (Polarization: " + pol + ", mode: %u, n: %u, nsub: %u)" % (mode,n,nsub)
		return 0
	
	mag = abs(R_data[poli,ni,mi,nsi,:])
	interp = interp1d(R_ds,mag,kind='cubic')

	R = interp(d)

	# check that beta exists for the intended waveguide/mode

	nclad = 1
	R = guidedOnly(R,n,nclad,d,mode,pol=pol,nsub=nsub)
	
	return R
	
def example_Ref():

	plt.figure(figsize=(6,5))
	
	n = 4
	
	xs = np.linspace(0.02,3,1000)

	for ni,n in enumerate([2,4,8]):
		ys = Ref(xs,pol='TM',n=n,mode=0,nsub=1)
	
		planar_r = (n-1)/(n+1)
	
		plt.plot(xs,ys,color=colors[ni],lw=2)
		plt.axhline(planar_r, color=colors[ni],linestyle='--')
		
	plt.ylim(0,1)
	plt.title('Reflection Magnitude')
	plt.xlabel(r'Height/$\lambda$')
	plt.ylabel('|R|')
	plt.show()
	
def example_Ref_high_Mx():

	data1 = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_TM_n_4_mode_1_nsub_1.txt',delimiter=',',dtype=complex)
	data2 = np.loadtxt('C:\\Users\\Patrick\\Documents\\PhD\\Data\\Rectangular Resonator\\reflection_coefficient_TM_n_4_mode_1_nsub_1_Mx_800.txt',delimiter=',',dtype=complex)

	R1 = abs(data1)
	R2 = abs(data2)
	
	p1 = np.angle(data1)/pi
	p2 = np.angle(data2)/pi
	
	ds1 = np.linspace(0.02,10,101)
	ds2 = R_ds
	
	plt.figure(figsize=(6,5))

	plt.plot(ds1,p1,'r')	
	plt.plot(ds2,p2,'b')	
	
	plt.ylim(0,1)
	plt.title('Reflection Magnitude')
	plt.xlabel(r'Height/$\lambda$')
	plt.ylabel('|R|')
	plt.show()

def example_Ref_vs_mode():

	plt.figure(figsize=(6,5))
	
	n = 4
	
	xs = np.linspace(0.02,3,1000)

	for mi,m in enumerate([0,1,2,3]):
		ys = Ref(xs,pol='TM',n=n,mode=m,nsub=1)
	
		planar_r = (n-1)/(n+1)
	
		plt.plot(xs,ys,color=colors[mi],lw=2)
		# plt.axhline(planar_r, color=colors[ni],linestyle='--')
		
	plt.ylim(0,1)
	plt.title('Reflection Magnitude (TM, n = 4)')
	plt.xlabel(r'Height/$\lambda$')
	plt.ylabel('|R|')

	leg = ['m = ' + str(x) for x in range(4)]
	plt.legend(leg,loc='best')
	plt.show()

	
def Finesse(d,pol='TM',m=1):
	
	R = Ref(d,pol=pol,mode=m)
	
	F = pi/(2*sp.arcsin((1-R)/(2*R**0.5)))
	F = np.nan_to_num(F)
	return F


def Beta(ncore,nclad,d,m,pol='TE',nsub=1,debug=False):
	if pol=='TE':m-=1
	'''
	returns real propogation constant of guided and leaky slab modes.  Normalized by wavelength (ie. returns Beta*wl instead of true Beta)
	
	Inputs:
	
	ncore, nclad, nsub - refractive indices of respective layers
	d - slab thickness normalized by high-index wavelength (d = hn/wl)
	m - mode order (fundamental = 1)
	pol - polarization using waveguides convention (transverse wrt propagation direction; TM = H out of plane)
	'''
	k = 2*pi	#note that wl is omitted since k*height = 2*pi*height/wl = 2*pi*d
	
	# Find Critical Angle and Brewster's Angles
	#convention: smaller number refers to smaller theta using Saleh's backwards theta convention
	tc1 = pi/2. - np.amax([sp.arcsin(nclad/ncore),sp.arcsin(nsub/ncore)])
	tc2 = pi/2. - np.amin([sp.arcsin(nclad/ncore),sp.arcsin(nsub/ncore)])
	
	tb1 = pi/2. - np.amax([sp.arctan(nclad/ncore),sp.arctan(nsub/ncore)])
	tb2 = pi/2. - np.amin([sp.arctan(nclad/ncore),sp.arctan(nsub/ncore)])
	
	# Compute phase pickup from reflections
	phi1 = lambda x: getPhi(ncore,nclad,sp.arcsin(x),pol=pol)
	phi2 = lambda x: getPhi(ncore,nsub ,sp.arcsin(x),pol=pol)
	
	# The Transcendental Equation
	F = lambda x: k*d*x - (m)*pi + phi1(x)/2. + phi2(x)/2. # x is sin(theta)
	
	sinths = np.ones(5) * np.nan	#list of sin(theta) values for guided + leaky modes (4 potential types of leaky mode)
	
	# check for zero crossings on a given interval - if present, input the value of sin(theta) into the array
	def getSinths(x0,x1,n):
		eps = 1e-15
		
		if x0 == x1: return 0
		
		
		try:
			sinths[n] = brentq(F,x0+eps,x1-eps)
		except ValueError: #no zero crossing present - ie no mode
			pass
			
			
	
	# all possible regions - note that two possibilities exist: tb1>tc2 or tb1<=tc2
	getSinths(1e-12,sin(tc1),0)
	getSinths(sin(tc1),sin(np.amin([tb1,tc2])),1)
	getSinths(sin(np.amin([tb1,tc2])),sin(np.amax([tb1,tc2])),2)
	getSinths(sin(np.amax([tb1,tc2])),sin(tb2),3)
	getSinths(sin(tb2),1,4)
	
	thetas = arcsin(sinths)
	betas  = ncore*2*pi*cos(thetas)
	
	if debug:
		st = np.linspace(0,1,3000)
		Fx = F(st)
		plt.plot(st,Fx/pi)
		plt.axhline(0,color='k')
		plt.axvline(sin(tb1),color='c',ls='--')
		plt.axvline(sin(tb2),color='m',ls='--')
		plt.axvline(sin(tc1),color='r',ls='--')
		plt.axvline(sin(tc2),color='y',ls='--')
		
		plt.xlabel(r'cos($\theta$)',fontsize=16)						# theta in this figure is relative to surface normal
		plt.ylabel(r'Dispersion / $\pi$',fontsize=16)
		
		print tc1,tc2
		print tb1,tb2
		print sin(tc1)
		print sinths
		plt.show()
		exit()


	return betas
	
	
def getPhi(ncore,nclad,theta,pol='TM',debug=False):
	'''
	Returns phase pickup due to internal reflection (core -> cladding).  Obtained from Fresnel reflection coefficient.
	'''
	e1 = ncore**2
	e2 = nclad**2
	
	R = putil.Reflection((pi/2. - theta),e1,e2,pol=pol,debug=debug)
	
	phi = np.angle(R)
	#if pol=='TE': phi *= -1
	if np.imag(nclad) != 0: phi *= -1
	
	return phi
	
def example_phi():
	
	ncore = 2
	nsub  = 1.5
	nclad = 1
	theta = np.linspace(0,pi/2,2000)
	angle = pi/2.-theta
	
	phic = getPhi(ncore,nclad,theta,pol='TM')
	phis = getPhi(ncore,nsub ,theta,pol='TM')
	
	plt.plot(angle*180/pi,phic/pi,'r-',lw=2)
	plt.plot(angle*180/pi,phis/pi,'r:',lw=2)
	plt.xlabel('Angle',fontsize=16)
	plt.ylabel(r'Phase Pickup, $\phi/\pi$',fontsize=16)
	plt.legend(('n = 1','n = 1.5'))
	
	plt.xlim(0,90)
	plt.ylim(0,1.1)
	
	plt.show()
	
def example_phi_metalsub():
	
	ncore = 4
	nsub  = 10j
	nclad = 1
	theta = np.linspace(0,pi/2,2000)
	angle = pi/2.-theta
	
	phic = getPhi(ncore,nclad,theta,pol='TM')
	phis = getPhi(ncore,nsub ,theta,pol='TM')
	
	plt.plot(angle*180/pi,phic/pi,'r-',lw=2)
	plt.plot(angle*180/pi,phis/pi,'r:',lw=2)
	plt.xlabel('Angle',fontsize=16)
	plt.ylabel(r'Phase Pickup, $\phi/\pi$',fontsize=16)
	plt.legend(('n = 1','n = 10j'))
	
	#plt.xlim(0,90)
	#plt.ylim(0,1.1)
	
	plt.show()
	
def getTheta(beta,ncore):
	'''
	back-calculate ray angle based on propagation constant.  
	Recall that for this module, theta is the COMPLIMENT of the angle of incidence.
	'''
	return sp.arccos(beta/(2*pi*ncore))
	
def example_theta():
	
	pol='TM'
	ncore = 2
	nclad = 1
	
	d = np.linspace(0.02,3,500) # in units of high-index wavelength
	mmax=3 						# highest order of waveguide mode to consider (m = # of field antinodes)
	
	# Beta values of each waveguide mode for each slab thickness.  0th index is slab height, 1st index is mode.
	Thetas = np.zeros((len(d),mmax))
	
	for i,di in enumerate(d):
		Bs = getBetas(ncore,nclad,di,pol=pol,mmax=mmax)
		Thetas[i,:]=90 - (getTheta(Bs,ncore) * 180/pi)
	
	
	for m in range(mmax):
		plt.plot(Thetas[:,m],d)
		
	tb = sp.arctan(1/ncore)
	plt.axvline(tb*180/pi,color='k',ls='--')
		
	tc = sp.arcsin(1/ncore)
	plt.axvline(tc*180/pi,color='y',ls='--')

	plt.xlim(0,90)
	
	plt.xlabel('Angle (degrees)')
	plt.ylabel(r'Height/$\lambda$')
	
	plt.show()
	
	
def getBetas(ncore,nclad,ds,ms,pol='TE',nsub=None):
	'''
	Returns 3d array of Betas.
	Indices (0,1,2) refer to (thickness, mode number, modal region)

	"modal region" is eg. guided, leaky in cover, leaky in sub, etc.
	
	INPUTS:
	ncore,nclad,nsub - refractive indices
	ds - thicknesses normalized by high-index wavelength
	ms - mode orders, 1 = fundamental
	pol - ['TE','TM']
	'''
	
	Bs = [Beta(ncore,nclad,d,m,pol,nsub=nsub) for d in ds for m in ms]
	Bs = np.array(Bs).reshape(len(ds),len(ms),5)	# 5 refers to different mode types (0 = guided, 1-4 = leaky)	
	
	return Bs	
	
def neff(B):
	'''
	converts values of Beta into mode indices
	'''
	
	wl = 2*sp.pi/B
	n = 1/wl
	return n

def example_neff():

	ncore = 4
	nclad = 1
	nsub  = 1
	
	ds = np.linspace(0.002,3,50)										# in units of high-index wavelength
	ms = np.arange(2) + 1				 								# order of waveguide mode to consider (m = # of field antinodes)
	
	Bs_TE = getBetas(ncore,nclad,ds,ms,pol='TE',nsub=nsub)
	Bs_TM = getBetas(ncore,nclad,ds,ms,pol='TM',nsub=nsub)
	ns_TE = neff(Bs_TE[:,0,0])
	ns_TM = neff(Bs_TM[:,0,0])
		
	plt.plot(ds,ns_TM,'r-',linewidth=2)
	plt.plot(ds,ns_TE,'b-',linewidth=2)
	plt.xlabel(r'Height / $\lambda$')
	plt.ylabel(r'$n_{eff}$')
	plt.title('Mode Index')
	plt.legend(('TM','TE'),loc='lower right')
	plt.show()
	

def rect_cavity_modes(ncore = 4,
  nclad = 1,
  nsub  = None,
  ds	= np.linspace(0.02,3,100),
  pol	= 'TM',
  ms	= np.array([1,2,3,4,5]),
  wmodes= 7,
  leaky = False,
  usePhase = True,
  linemap   = False,
  colormap  = True
  ):

	'''
	Plots map of resonant locations based on resonator width and height (each normalized by high-index wavelength.
	
	TODO - eliminate gap between guided and leaky mode curves
	
	INPUTS:
	ncore,nclad,nsub: indices of refraction.  In nsub == None, nsub=nclad
	ds: slab thickness in units of high-index wavelength
	pol: ['TE','TM'] 
	mmax: highest mode order to plot (1 = fundamental)
	wmodes: number of lateral resonances to display (with no phase, the 1st will be a line at width=0)
	leaky: if true, plot leaky modes
	phase: if true, include phase pickup when computing resonant widths.  Else, phi=0
	linemap: displays line plot of resonances
	colormap: displays colormap plots of resonances modulated by Fabry-Perot Quality Factor
	
	RETURNS:
	ds: slab thicknesses used in computation
	w : 4D matrix of resonant widths [slab thickness d, lateral mode #, waveguide mode order m, mode type 0-5]
	
	'''
							
	if nsub==None: nsub=nclad

	# Get beta values
	Bs  = getBetas(ncore,nclad,ds,ms,pol=pol,nsub=nsub)
	Bs /= ncore
	if not leaky:
		for i in (1,2,3,4): Bs[:,:,i] = np.nan
	
	# w is a 4D matrix containing the resonant width for each slab thickness d, lateral mode, waveguide mode order, and mode type.
	# Indices are in the respective order.
	w = np.zeros((len(ds),wmodes,len(ms),5),dtype=float)
	
	# Compute lateral mode positions
	for mi,m in enumerate(ms-1):
		
		if not usePhase: phi = 0 #-1*pi/2.
		else: phi = phase(ds,pol=pol,mode=m,nsub=nsub)*pi/180. 
		'''
		#phi = usePhase
		
		phi1 = phase(ds,pol=pol,mode=m,nsub=1)*pi/180.
		phi2 = phase(ds,pol=pol,mode=m,nsub=2)*pi/180.
		'''
		for i in range(wmodes):
			for j in range(5):
				w[:,i,mi,j] = ((i+1)*pi-(phi+phi)/2.)/Bs[:,mi,j]					#(i+1) is used to bypass the 0th order lateral resonance that is just a constant line at w=0
	
	# Display data
	if linemap: line_plot(ds,w)
	if colormap: colormap_plot(ncore,ds,w,Bs,pol=pol)
	
	if linemap or colormap: plt.show()
	
	return ds,w
	
def effect_of_index():

	pol='TM'
	ncore = [2,4,8]
	nclad = 1
	
	d = np.linspace(0.02,3,500) # in units of high-index wavelength
	mmax=1 						# highest order of waveguide mode to consider (m = # of field antinodes)
	wmodes=2 					# number of lateral resonances to consider (with no phase, the 1st will be a line at width=0)
	
	# Beta values of each waveguide mode for each slab thickness.  0th index is slab height, 1st index is mode, 2nd is core index
	BB = np.zeros((len(d),mmax,len(ncore)))
	
	for i,di in enumerate(d):
		for nci, nc in enumerate(ncore):
			Bs = getBetas(nc,nclad,di,pol=pol,mmax=mmax)
			Bs /= nc # convert from free-space wavelengths to high-index wavelengths
			
			BB[i,:,nci]=Bs
	
	# w is a 3D matrix containing the resonant width for each slab thickness d, lateral mode, and waveguide mode order.  Indices are in the respective order.
	w = np.zeros((len(d),wmodes,mmax,len(ncore)))
	
	for m in range(mmax):
		phi = 0 #phase(d,pol=pol,mode=m,n=ncore)*pi/180. 
		for i in range(wmodes):
			for nci,nc in enumerate(ncore):
				w[:,i,m,nci] = (i*pi-phi)/BB[:,m,nci]
	
	# figure in units of high-index wavelength
	plt.figure(figsize=(5,5))
	for i in range(len(w[0,:,0,0])):
		for j in range(len(w[0,0,:,0])):
			for n in range(len(w[0,0,0,:])):
				plt.plot(w[:,i,j,n],d,color=colors[n])	
	
	plt.xlim(0.02,3)
	plt.ylim(0.02,3)
	plt.xlabel(r'Width / $\lambda$')	
	plt.ylabel(r'Height / $\lambda$')	
	plt.title('Truncated Dielectric Slab Resonances')
	plt.legend(np.array(ncore).astype(str))
	
	# Renormalized figure into units of free-space wavelength
	plt.figure(figsize=(5,5))
	for i in range(len(w[0,:,0,0])):
		for j in range(len(w[0,0,:,0])):
			for n in range(len(w[0,0,0,:])):
				plt.plot(w[:,i,j,n]/ncore[n],d/ncore[n],color=colors[n])	
	
	plt.xlim(0.02,3)
	plt.ylim(0.02,1)
	plt.xlabel(r'Width / $\lambda_{o}$')	
	plt.ylabel(r'Height / $\lambda_{o}$')	
	plt.title('Truncated Dielectric Slab Resonances')
	plt.legend(np.array(ncore).astype(str))
	plt.show()

def line_plot(d,w):
	
	fig = plt.figure(figsize=(5,5))
	ax  = fig.add_subplot(111)
	
	for i in range(len(w[0,:,0,0])):
		for j in range(len(w[0,0,:,0])):
			for k in range(5):
				ax.plot(w[:,i,j,k],d,color=colors[j])
	
	ax.set_xlim(0.02,3)
	ax.set_ylim(0.02,3)
	ax.set_xlabel(r'Width / $\lambda$')	
	ax.set_ylabel(r'Height / $\lambda$')	
	ax.set_title('Truncated Dielectric Slab Resonances')

	return 0
		
def colormap_plot(ncore,d,w,Bs,pol='TM'):
	
	cmap = np.zeros((len(d),len(d),len(w[0,0,:,0])))
	
	for j in range(len(w[0,0,:,0])): # j is mode order

		# carry over previous modes
		if j>0: cmap[:,:,j] += cmap[:,:,j-1]

		for i in range(len(w[0,:,0,0])): # i is FP mode order
			
			
			# convert width values to index locations in d
			line = w[:,i,j,0]/d[-1]*len(d)								# rescale 0-dmax to 0-# of d points
			line = np.rint(line).astype(int)							# convert values to integers
			line = line * (0 < line) * (line < len(d))					# get rid of anything that is not a valid index

			di = np.arange(len(d))
			
			'''
			print w[:,i,j]
			print d[-1]
			print line
			print di
			'''
			
			neffs = neff(Bs)
			
			# Assign values in the matrix corresponding to the calculated FP Quality Factor (see notebook pg 19 for derivation)
			
			# normalize for angular incidence
			t = 0														# t = 0 is normal incidence
			
			angle = (t+90) * pi/180.
			wl = 400
			h  = d		   *wl/ncore
			wi = w[:,i,j,0]*wl/ncore
			m = i+1
			n = j+1
			
			FF_v = DI.getFF(wl,wi,h,m,n,angle)
			FF_h = DI.getFF(wl,wi,h,m,n,angle+pi/2.)
			
			cmap[line,di,j] += Finesse(d,pol=pol,m=j)*2*w[:,i,j,0]*neffs[:,j,0]/ncore * FF_h
			cmap[di,line,j] += Finesse(d,pol=pol,m=j)*2*w[:,i,j,0]*neffs[:,j,0]/ncore * FF_v
			
			cmap[:,:,j] = np.nan_to_num(cmap[:,:,j])
			
			
			# Clean up noise at edges
			for n in range(10):
				cmap[:,n,j] = 0
				cmap[n,:,j] = 0
			
			print j,i,np.amax(cmap[:,:,j]),np.where(cmap[:,:,j]==np.amax(cmap[:,:,j]))
			#cmap += vert + horiz
			
		
		
		# blur lines for aesthetic purposes
		#cmap_blur = blur_image(cmap[:,:,j],10)
		
		plt.figure(figsize=(6.5,5))		
		
		#plt.imshow(cmap[:,:,j],vmin=0,vmax=None, origin='lower')
		plt.imshow(cmap[:,:,j],extent=(d[0],d[-1],d[0],d[-1]),vmin=0,vmax=None, origin='lower')
		#plt.imshow(cmap_blur,extent=(d[0],d[-1],d[0],d[-1]),vmin=0,vmax=None, origin='lower')
		
		plt.colorbar()
		plt.xlabel(r'Width / $\lambda$')	
		plt.ylabel(r'Height / $\lambda$')	
		plt.title('Fabry-Perot Quality Factor')

	return 0

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = sp.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = convolve(im,g, mode='valid')
    return(improc)
	
def test_Beta_simple():
	
	pol='TE'
	ncore = 2
	nclad = 1
	nsub  = 1
	
	d = 0.2						# in units of high-index wavelength
	m = 1 						# highest order of waveguide mode to consider (m = # of field antinodes)
	
	Beta(ncore,nclad,d,m,pol,nsub,debug=True)

def test_Beta():
	
	pol='TM'
	ncore = 2
	nclad = 1
	nsub  = 1.5
	
	ds = np.linspace(0.02,3,500)						# in units of high-index wavelength
	ms = [1,2,3]				 						# order of waveguide mode to consider (m = # of field antinodes)
	
	Bs = [Beta(ncore,nclad,d,m,pol,nsub) for d in ds for m in ms]
	Bs = np.array(Bs).reshape(len(ds),len(ms),5)		# indices (0,1,2) refer to (thickness, mode number, modal region)
	
	colors = ['r','b','g','m','c']
	styles = ['-','--',':','-','--']
	
	for r in range(5):
		for m in range(len(ms)):
			plt.plot(Bs[:,m,r],ds,color=colors[m],ls=styles[r])
	
	plt.show()
	
def example_wk():

	pol='TM'
	ncore = 2
	nclad = 1
	nsub  = 1
	
	ds = np.linspace(0.02,3,1000)										# in units of high-index wavelength
	ms = np.arange(5) + 1				 								# order of waveguide mode to consider (m = # of field antinodes)
	
	# get w from d, supposing a structure with height h:
	h = 100 * 1e-9 #m
	wls = h*ncore/ds
	c = 3e8
	ws = 2*pi*c/wls
	
	Bs = [Beta(ncore,nclad,d,m,pol,nsub) for d in ds for m in ms]
	Bs = np.array(Bs).reshape(len(ds),len(ms),5)		# indices (0,1,2) refer to (thickness, mode number, modal region)
	
	# add light lines
	
	plt.figure(figsize=(6.5,5))
	
	#light lines for critical angles
	plt.plot(ws/c*nclad,ws,'k-',lw=2)
	plt.plot(ws/c*nsub ,ws,'k-',lw=2)
	plt.plot(ws/c*ncore,ws,'k-',lw=2)
	
	#light lines for brewsters angles
	# tbc = arctan(nclad/ncore)
	# tbs = arctan(nsub /ncore)
	tbc = arctan(ncore/nclad)
	tbs = arctan(ncore/nsub)
	
	plt.plot(ws/c*sin(tbc)*nclad,ws,'k:',lw=2)
	plt.plot(ws/c*sin(tbs)*nsub,ws,'k:',lw=2)

	colors = ['r','#00B900','b','#FBC715','m']
	styles = ['-','--',':','-','--']
	
	for r in range(5):
		for m in range(len(ms)):
			plt.plot(Bs[:,m,r]/wls,ws,color=colors[m],lw=2)
			
	plt.xlim(0,2e8)
	plt.ylim(0,np.amax(ws))
	
	plt.xlabel(r'$k_{x}$',fontsize=16)
	plt.ylabel(r'$\omega$',fontsize=16)
	plt.title('Core = 2; Substrate Index = 1',fontsize=16)
			
	plt.show()
	
def Attenuation(d,ncore,nclad,nsub,angle,pol='TM'):
	'''
	Returns the attenuation coefficient alpha (imaginary part of the propagation constant)
	as calculated according to "Attenuation of Cutoff modes and Leaky modes in Dielectric Slab
	Structures." D. Miller and H. Haus, Bell Labs 1986
	
	alpha is returned in dimensionless units - divide by wavelength for true value
	'''
	
	e1  = ncore**2
	e2c = nclad**2
	e2s = nsub**2
	
	rc = putil.Reflection(angle,e1,e2c,pol=pol)
	rs = putil.Reflection(angle,e1,e2s,pol=pol)
	
	alpha = 2/(tan(angle)*d)*(1-rs*rc)/(sqrt(rs*rc))
	
	return alpha
	
def example_Attenuation():
	
	ncore = 2
	nclad = 1
	nsub  = 1
	
	d     = np.linspace(0.02,3,100)
	angle = np.linspace(0.02,pi/2.,100)
	
	tc = arcsin(nclad/ncore)
	
	a = np.array([Attenuation(x,ncore,nclad,nsub,y,pol='TM') for x in d for y in angle])
	
	a = a.reshape(len(d),len(angle))
	
	ext = [np.amin(angle)*180/pi,np.amax(angle)*180/pi,np.amin(d),np.amax(d)]
	plt.imshow(a.real,interpolation='nearest',origin='lower',extent=ext,aspect='auto',vmax=2000)
	plt.colorbar()
	plt.axvline(tc*180/pi,color='w',ls=':')
	
	plt.figure()
	plt.imshow(a.imag,interpolation='nearest',origin='lower',extent=ext,aspect='auto',vmax=2000)
	plt.colorbar()

	plt.show()

def example_Attenuation_single():
	
	ncore = 2
	nclad = 1
	nsub  = 1
	
	d     = 0.02
	angle = 0.02
	
	print Attenuation(d,ncore,nclad,nsub,angle,pol='TM')
	
def overlay(modes='horizontal',asym=False,noLines=True):
	'''
	Display cavity mode map over FDFD results
	
	NOTES:
	
	- There is a reversal in polarization convention between FDFD and waveguides
	'''
	
	Q_TE_n4 	  = 'parameter sweeps/Q_TE_n4_45deg.pickle'
	Q_TM_n4 	  = 'parameter sweeps/Q_TM_n4_45deg.pickle'
	Q_TE_n4_0_deg = 'parameter sweeps/Q_TE_n4_0deg.pickle'
	Q_TE_n4_lossy = 'parameter sweeps/lossy/Q_TE_n4_k0_1_45deg.pickle'
	Qabs_TE_n4_lossy = 'parameter sweeps/lossy/Qabs_TE_n4_k0_1_45deg.pickle'
	Q_TE_n4_nsub2 = 'parameter sweeps/w_substrate/Q_TE_n4_nsub2_45deg.pickle'
	Q_TE_n4_metalsub = 'parameter sweeps/w_substrate/Q_TE_n4_esub-100_45deg.pickle'
	Q_TE_n4_metalsides = 'parameter sweeps/sidewalls/Q_TE_n4_nsides10j_wref_100.pickle'
	Q_TM_n4_metalsides = 'parameter sweeps/sidewalls/Q_TM_n4_nsides10j_wref_100.pickle'
	
	# Load FDFD data
	with open(PATH + Q_TM_n4,'rb') as f:
		fdfd = pickle.load(f)
		
	ds = np.linspace(0.02,3.,100)
	
	if not noLines:	
		# Generate resonance map
		ds,ws = rect_cavity_modes(ncore=4,
								  nclad=1,
								  nsub =1,
								  ms   =np.array([3]),
								  wmodes=7,
								  pol  ='TM',
								  linemap=False,
								  colormap=False,
								  usePhase=True#0.5*(pi+phase(ds,pol='TM',mode=0,nsub=1))*pi/180. 
								  )
		if asym and not (modes=='horizontal' or modes=='vertical'): #do vertical modes						  
			ds,wsv = rect_cavity_modes(ncore=4,
									  nclad=1,
									  nsub =1,
									  ms   =np.array([1]),
									  wmodes=7,
									  pol  ='TM',
									  linemap=False,
									  colormap=False,
									  usePhase=True
									  )
		else: wsv = ws
		
	fig = plt.figure(figsize=(6.5,5))
	ax  = fig.add_subplot(111)
	
	ext = putil.getExtent(ds,ds)
	
	vmax = 50
	vmin = 0
	im = ax.imshow(fdfd,extent=ext,vmin=vmin,vmax=vmax)
	fig.colorbar(im)
	
	if noLines:
		
		ax.set_xlabel(r'Width / $\lambda$')	
		ax.set_ylabel(r'Height / $\lambda$')	
		ax.set_title('Truncated Dielectric Slab Resonances')
		plt.show()
		exit()
		
	for i in range(len(ws[0,:,0,0])):
		for j in range(len(ws[0,0,:,0])):
			for k in range(5):
				if modes == 'horizontal':
					if i%2==0: ax.plot(ws[:,i,j,k],ds,color='w',lw=3,ls='-')
					if i%2==1: ax.plot(ws[:,i,j,k],ds,color='w',lw=3,ls=':')
				if modes == 'vertical'  : ax.plot(ds,ws[:,i,j,k],color='w',lw=3,ls='-')
				if modes == 'both'      :
					ax.plot(ds,wsv[:,i,j,k],color='w',lw=2,ls='-') #vert
					ax.plot(ws[:,i,j,k] ,ds,color='w',lw=2,ls='-') #horiz
				
	ax.set_xlim(0.02,3)
	ax.set_ylim(0.02,3)
	ax.set_xlabel(r'Width / $\lambda$')	
	ax.set_ylabel(r'Height / $\lambda$')	
	ax.set_title('Truncated Dielectric Slab Resonances')
	#ax.set_title(r'$Q_{abs}$',fontsize=18)
							  
	
	plt.show()
	
def effective_index_method(ncore=4,
	nclad=1,
	nsub=1,
	ds=np.linspace(0.02,3,100),
	ms=np.arange(1) + 1,
	pol='TE'
	):

	'''
	Computes reflection coefficient going from a layer having index neff to a layer of nclad.
	Used to approximate Q and phase pickup in resonators.
	'''
	
	Bs = getBetas(ncore,nclad,ds,ms,pol=pol,nsub=nsub)
	
	neffs = neff(Bs)
	
	e1 = neffs**2
	e2 = nclad**2
	
	Rs = putil.Reflection(0,e1,e2,pol=pol)
	
	r   = abs(Rs)
	phi = np.angle(Rs)
	
	return r,phi
	
def example_effective_index_method():
	
	ds = np.linspace(0.02,3,100)
	ms = np.arange(5) + 1
	
	r,phi = effective_index_method(ds=ds,ms=ms)
	
	fig,axs=plt.subplots(2,1)
	
	for m in ms-1:
		axs[0].plot(ds,r[:,m,0],  'r',lw=2)
		axs[1].plot(ds,phi[:,m,0],'b',lw=2)
		
	axs[0].set_ylabel('|R|')
	axs[0].set_ylim(0,1)
	
	axs[1].set_ylabel(r'$\phi/\pi$')
	
	plt.show()
		
def compare_rcwa_effective_index():
	
	ds = np.linspace(0.02,3,100)
	m  = np.array([1])
	
	pol = 'TM'
	
	r,phi = effective_index_method(ds=ds,ms=m,pol=pol)
	
	rcwaRef = Ref(  ds,pol=pol,mode=0,n=4)
	rcwaPhi = phase(ds,pol=pol,mode=0,n=4) / 180.
	
	fig,axs=plt.subplots(2,1,figsize=(6.5,5))
	
	
	axs[0].plot(ds,r[:,0,0],  'r:',lw=2)
	axs[0].plot(ds,rcwaRef ,  'r' ,lw=2)
	
	
	axs[1].plot(ds,phi[:,0,0],'b:',lw=2)
	axs[1].plot(ds,rcwaPhi,   'b' ,lw=2)
		
	axs[0].set_ylabel('|R|',fontsize=16)
	axs[0].set_ylim(0,1)
	
	axs[1].set_ylabel(r'$\phi/\pi$',fontsize=16)
	axs[1].set_xlabel(r'$d/\lambda$',fontsize=16)
	axs[1].legend(('Effective Index','RCWA'))
	axs[1].set_ylim(0,0.5)
	
	plt.show()
	
def effect_of_substrate():
	
	nsubs = np.array([1,2,3,-10j])

	fig = plt.figure(figsize=(5,5))
	ax = fig.add_subplot(111)
	
	for nsubi,nsub in enumerate(nsubs):
		print nsub
		
		# Generate resonance map
		ds,ws = rect_cavity_modes(ncore=4,
								  nclad=1,
								  nsub =nsub,
								  ms=np.array([1]),
								  wmodes=1,
								  pol  ='TM',
								  linemap=False,
								  usePhase=False
								  )


		for i in range(len(ws[0,:,0,0])):
			for j in range(len(ws[0,0,:,0])):
				ax.plot(ws[:,i,j,0],ds,color=colors[nsubi],lw=2)
					

	ax.set_xlim(0.02,3)
	ax.set_ylim(0.02,3)
	ax.set_xlabel(r'Width / $\lambda$')	
	ax.set_ylabel(r'Height / $\lambda$')	
	ax.set_title(r'Resonant Position vs $n_{sub}$')
	ax.legend((r'$n_{sub}$ = 1',r'$n_{sub}$ = 2',r'$n_{sub}$ = 3',r'$n_{sub}$ = 10j'))
	
	plt.show()
								  
"""
Obselete Functions

def getBetas(ncore,nclad,d,pol='TE',mmax=None,nsub=None):
	'''
	Returns list of propagation constants for a dielectric slab waveguide, thickness d in units of wls (TE modes).
	'''

	M = numTotalModes(ncore,nclad,d)
	
	print "d: %0.2f; Number of Supported Modes: %u" % (d,M)

	Bs = []
	
	if mmax==None: l=M
	else: l=mmax

	for m in range(l):
		m += 1
		if m>M:
			Bs.append(np.nan)
		else:
			if nsub == None: Bs.append(Beta(ncore,nclad,d,m,pol,nsub=nclad)[0])
			else: 			 Bs.append(Beta(ncore,nclad,d,m,pol,nsub=nsub )[0])
					
	return np.array(Bs)
	

def Beta(ncore,nclad,d,m,pol='TE'):
	'''
	Returns propagation constant of the mth mode in a dielectric slab waveguide, thickness d in units of wls.
	'''
	
	critical_angle = sp.arcsin(nclad/ncore)
	
	if m/(2*d) > sp.sin(pi/2. - critical_angle): #leaky modes
		print 'Warning: Leaky mode at m =',m	
		if pol=='TE':
			theta = sp.arcsin(m/(2*d))
		else:
			theta = sp.arcsin(m/(2*d)) #needs to be corrected to include brewster's angle behavior
	
	elif m/(2*d) <= sp.sin(pi/2. - critical_angle): #bound modes
		
		if pol=='TE':
			F = lambda x: sp.tan(pi*d*x - m*pi/2.) - sp.sqrt(sp.sin(pi/2.-critical_angle)**2 / (x**2) - 1) # x is sin(theta)
		else:
			F = lambda x: sp.tan(pi*d*x - m*pi/2.) - 1/sp.cos(pi/2.-critical_angle)**2 * sp.sqrt(sp.sin(pi/2.-critical_angle)**2 / (x**2) - 1)
		
		x0 = m/(2*d)
		x1 = (m+1)/(2*d)-1e-12
		
		'''
		xs = np.linspace(x0,x1,1000)
		ys = map(F,xs)
		plt.plot(xs,ys)
		plt.ylim(-5,5)
		plt.show()
		exit()
		'''
		
		sinth = brentq(F,x0,x1)
		
		theta = sp.arcsin(sinth)
		
	beta = ncore*2*pi*sp.cos(theta)
		
	return beta


def dispersion_TM(beta,ncore,nclad,d,m):
	'''
	from Inan Electromagnetic Waves pg 291, eq 4.40
	'''
	
	k = ncore    #k = ko*n, normalized by ko
	bx = sqrt(k**2 - beta**2)
	
	# a is transverse wavevector
	a = lambda b: sqrt((ncore**2-nclad**2)-b**2)
	
	return a(bx)/bx, (nclad/ncore)**2 * sp.tan(2*pi/ncore * bx*d/2. - m*pi/2.), a(bx)/bx - (nclad/ncore)**2 * sp.tan(2*pi/ncore * bx*d/2. - m*pi/2.)    #the first factor of 2pi/n in the tan() is to compensate for normalization

def example_dispersion():
	
	ncore = 4
	nclad = 1
	
	d = 0.2*ncore
	
	kmax = ncore
	re = np.linspace(0,1,1000)
	im = np.linspace(0,kmax*2,1000)
	
	B_re, B_im = np.meshgrid(re,im)

	B = B_re + 1j*B_im
	
	disp_odd = abs(dispersion_TM(B,ncore,nclad,d,0)[2])
	disp_even = abs(dispersion_TM(B,ncore,nclad,d,1)[2])
	disp = disp_odd*disp_even
	
	plt.imshow(sp.log10(disp),origin='lower',extent=[0,kmax,-kmax/2,kmax/2],vmax=2,cmap='jet',interpolation='nearest')
	plt.colorbar()
	plt.show()
	
def debug():
	
	ncore = sqrt(2)
	nclad = 1
	
	d = 1.25*ncore
	
	B = sp.linspace(0,2,100)
	
	ab,t,x = dispersion_TM(B,ncore,nclad,d,0)

	plt.plot(B,ab,'r-')
	plt.plot(B,t,'b-')
	plt.ylim(-1,1)
	plt.show()

def example_dispersion_Inan():
	ncore = sp.sqrt(2)
	nclad = 1
	
	d = 1.25
	
	bx = np.linspace(0,2*pi*2/d,500)
	a = lambda b: sp.sqrt((2*pi)**2*(ncore**2-nclad**2)-b**2)
	
	ax = np.array(map(a,bx))
	t = nclad**2/ncore**2 * sp.tan(bx*d/2)
	t2 = nclad**2/ncore**2 * sp.tan(bx*d/2+pi/2.)
	
	plt.figure(figsize=(5,5))
	plt.plot(bx*d/2./pi,ax*d/2./pi,'r-')
	plt.plot(bx*d/2./pi,t/pi,'b-')
	plt.plot(bx*d/2./pi,t2/pi,'b-')
	
	for i in range(4):
		plt.axvline(1/2*i,color='k',linestyle='--')
		
	plt.xlim(0,2)
	plt.ylim(0,2)
	plt.show()
	
	
def effect_of_substrate():

	pol='TM'
	ncore = 2
	nclad = 1
	nsub = [1,2,3]
	
	d = np.linspace(0.02,3,50) # in units of high-index wavelength
	mmax=2 						# highest order of waveguide mode to consider (m = # of field antinodes)
	wmodes=2 					# number of lateral resonances to consider (with no phase, the 1st will be a line at width=0)
	
	# Beta values of each waveguide mode for each slab thickness.  0th index is slab height, 1st index is mode, 2nd is core index
	BB = np.zeros((len(d),mmax,len(nsub)))
	
	for i,di in enumerate(d):
		for nsi, ns in enumerate(nsub):
			Bs = getBetas(ncore,nclad,di,pol=pol,mmax=mmax,nsub=ns)
			Bs /= ncore # convert from free-space wavelengths to high-index wavelengths
			
			BB[i,:,nsi]=Bs
	
	# w is a 3D matrix containing the resonant width for each slab thickness d, lateral mode, and waveguide mode order.  Indices are in the respective order.
	w = np.zeros((len(d),wmodes,mmax,len(nsub)))
	
	for m in range(mmax):
		phi = 0 #phase(d,pol=pol,mode=m,n=ncore)*pi/180. 
		for i in range(wmodes):
			for nsi,ns in enumerate(nsub):
				w[:,i,m,nsi] = (i*pi-phi)/BB[:,m,nsi]
	
	# figure in units of high-index wavelength
	plt.figure(figsize=(5,5))
	for i in range(len(w[0,:,0,0])):
		for j in range(len(w[0,0,:,0])):
			for n in range(len(w[0,0,0,:])):
				plt.plot(w[:,i,j,n],d,color=colors[n])	
	
	plt.xlim(0.02,3)
	plt.ylim(0.02,3)
	plt.xlabel(r'Width / $\lambda$')	
	plt.ylabel(r'Height / $\lambda$')	
	plt.title('Truncated Dielectric Slab Resonances')
	plt.legend(np.array(ncore).astype(str))
	
	# Renormalized figure into units of free-space wavelength
	plt.figure(figsize=(5,5))
	for i in range(len(w[0,:,0,0])):
		for j in range(len(w[0,0,:,0])):
			for n in range(len(w[0,0,0,:])):
				plt.plot(w[:,i,j,n]/ncore[n],d/ncore[n],color=colors[n])	
	
	plt.xlim(0.02,3)
	plt.ylim(0.02,1)
	plt.xlabel(r'Width / $\lambda_{o}$')	
	plt.ylabel(r'Height / $\lambda_{o}$')	
	plt.title('Truncated Dielectric Slab Resonances')
	plt.legend(np.array(ncore).astype(str))
	plt.show()
	
	
	


"""


if __name__ == '__main__':
	#rect_cavity_modes()
	#effect_of_index()
	#effect_of_substrate()
	# example_neff()
	# example_Ref()
	#example_phase()
	#example_theta()
	#example_phi()
	#example_phi_metalsub()
	#test_Beta_simple()
	#test_Beta()
	#example_wk()
	#example_Attenuation_single()
	#example_Attenuation()
	# overlay()
	#example_effective_index_method()
	#compare_rcwa_effective_index()
	# effect_of_mirrors_on_phase()
	# effect_of_mirrors_on_R()
	# example_Ref_high_Mx()
	example_Ref_vs_mode()
	# example_phase_vs_mode()