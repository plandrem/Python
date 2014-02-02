#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

import DielectricSlab as DS
import DielectricCylinderDispersion as DCD

pi = sp.pi
sqrt = sp.emath.sqrt

def SideCoupledDisk_2D():
	
	# waveguide width (nm)
	w = 130
	
	# design wavelength (nm)
	wl = 1700
	
	#get propagation constant for waveguide
	ncore = 4
	nclad = 1
	nsub  = 1
	
	d = w*ncore/wl
	
	pol = 'TM'
	
	beta = DS.Beta2(ncore,nclad,d,1,pol,nsub)[0]
	beta /= wl	#restore to absolute units
	
	ko = 2*pi/wl
	neff = beta/ko
	
	# choose target LMR order
	m = 3
	
	# calculate corresponding radius
	r = (wl/neff)*m/(2*pi)
	
	print r
	
	
	return 0
	
def waveguide_modeindex(widths,wl,ncore,pol = 'TM'):
	
	#get propagation constant for waveguide
	nclad = 1
	nsub  = 1
	
	d = widths*ncore/wl
	
	betas = np.array([DS.Beta2(ncore,nclad,di,1,pol,nsub)[0] for di in d])
	betas /= wl	#restore to absolute units
	
	ko = 2*pi/wl
	neff = betas/ko
	
	return neff
	
def cavity_modeindex(rads,wl,n,pol='TM',jmax=4,mmax=4):
	   
	wl   *= 1e-9
	rads *= 1e-9
	resrad = DCD.getResonantRadii(wl,rads,n,pol,jmax,mmax)
	
	neff = np.zeros((jmax,mmax))
	
	for j in range(jmax):
		neff[j,:] = DCD.effectiveIndex(wl,resrad[j,:],j)
		
	return resrad,neff

def side_coupled_disk_dispersion():
	widths = np.linspace(50,400,100)
	rads   = np.linspace(50,1000,500)
	wl = 1500
	n = 4
	
	pol = 'TM'
	
	jmax=10
	mmax=4

	wg_neff = waveguide_modeindex(widths,wl,n,pol='TM')	
	resrad,cv_neff = cavity_modeindex(rads,wl,n,pol,jmax,mmax)	
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax2 = ax.twinx()
	
	ax.plot(wg_neff,widths)
	
	for j in range(jmax):
		ax2.plot(cv_neff[j,:],resrad[j,:]*1e9,ls='',marker='o')
		
	ax.set_ylabel('Waveguide Width')
	ax2.set_ylabel('Cavity Radius')
	
	ax.set_xlabel(r'$n_{eff}$')
		
	plt.show()
	
def side_coupled_rect():
	
	# get effective index in waveguide
	wl = 1500
	h = 250
	ncore = 3.48
	d = h*ncore/wl
	
	neff_wg = DS.neff(DS.Beta2(ncore,1,d,1,'TM')[0])
	
	# plot effective index of cavity waveguide
	ds = np.linspace(0.02,3,100)
	ns = np.zeros(len(ds)) * np.nan
	
	for i,di in enumerate(ds):
		n = DS.neff(DS.Beta2(4.1,1,di,1,pol='TM')[0])
		ns[i] = n
		
	plt.plot(ds,ns,'r-',linewidth=2)
	plt.axhline(neff_wg,color='k',linestyle=':',linewidth=2)
	plt.xlabel(r'Height / $\lambda$')
	plt.ylabel(r'$n_{eff}$')
	plt.title('Mode Index')
	plt.show()

def getFPdims():
	
	d = 0.542
	
	h = d*1500/4.1
	
	beta = DS.Beta2(4.1,1,d,1,pol='TM')[0]/1500
	
	m = 4
	
	phi = DS.phase(d,'TM',0,4) * pi/180.
	print 'beta;',beta
	print 'phi;',phi
	
	length = (m*pi-phi)/beta
	
	print 'height:',h
	print 'length:',length

if __name__ == '__main__':
	#side_coupled_rect()
	getFPdims()
	
