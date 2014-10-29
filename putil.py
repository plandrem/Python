#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl


import sys
import os
import time

from scipy import cos, sin, pi
from scipy.stats import linregress
from scipy.interpolate import interp1d

PATH = '/Users/Patrick/Documents/PhD/'
DATA_PATH = PATH + '/DATA/'

'''
Constants
'''

h = 6.62606957e-34
c = 299792458
q = 1.602176565e-19

'''
Color scales for line plots
'''
greens = ['k','#106B00','#148700','#19A800','#1FCC00','#24F000','#77F062','#B0F2A5']

class LinReg:
	'''
	Given lists of x and y data, produces a linear regression and returns new values along the regression line.
	xmin,xmax restrict the domain of input values to be considered for the regression
	'''
	def __init__(self,x,y,xmin=None,xmax=None):
		
		pairs = zip(x,y)
		f = lambda (x,y): (x >= xmin or xmin == None) and (x <= xmax or xmax == None)

		filtered = np.array(filter(f,pairs))
		xs = filtered[:,0]
		ys = filtered[:,1]

		m,b=linregress(xs,ys)[0:2]
		self.m = m
		self.b = b
		
	def __call__(self,x):
		'''return calculated y value for new x input'''
		return self.m*x + self.b
		
	def get_xint(self):
		'''return x intercept'''
		return -1*self.b/self.m
		
		
def maplist(func,var,*args):
	'''
	Applies the built in map() function to a function which requires multiple arguments.  The first
	argument is treated as a dependent variable, and is given by the list "var".  The remaining 
	arguments, which remain constant for all iterations, are entered subsequently in *args.
	'''
	
	l = len(var)
	arglist = []
	
	for arg in args:
		a = [arg for i in range(l)]
		arglist.append(a)
		
	return map(func,var,*arglist)
	
def Reflection(angle,e1,e2,pol='TE',debug=False):
	'''
	Return complex reflection coefficient for two dielectric interface.  Magnitude is abs(R), phase is np.angle(R)
	
	Sign convention taken from Saleh and Teich Fundamentals of Photonics, because it makes things work in
	DielectricSlab.py
	
	'''
	
	sec = lambda x: 1/cos(x)
	
	n1 = sp.emath.sqrt(e1)
	n2 = sp.emath.sqrt(e2)
	
	z1 = 1/n1
	z2 = 1/n2
	
	#print 'putil.Reflection:'
	#print angle
	
	angle2 = sp.arcsin((n1/n2)*sin(angle))
	#print angle2
	
	# for some reason, arcsin(sin) introduces some weird jitter into the imaginary part
	angle2 = angle2.real + -1j*abs(angle2.imag)
		
	if   pol=='TE': R = (z2*sec(angle2)-z1*sec(angle))/(z1*sec(angle)+z2*sec(angle2))
	elif pol=='TM': R = (z2*cos(angle2)-z1*cos(angle))/(z1*cos(angle)+z2*cos(angle2))
	
	else:
		print 'putil.Reflection : Invalid Polarization.'
		return -1
	
	R = R * (abs(R.imag) >= 1e-12) + R.real * (abs(R.imag) < 1e-12)
	
	if debug:
		
		plt.plot(angle,R.real,'r')
		plt.plot(angle,R.imag,'b')
		'''
		plt.figure()
		plt.plot(angle/pi,(-1*z1*cos(angle)+z2*cos(angle2)).real,'r-')
		plt.plot(angle/pi,(-1*z1*cos(angle)+z2*cos(angle2)).imag,'r--')
		plt.plot(angle2/pi,((z1*cos(angle)+z2*cos(angle2))).real,'b-')
		plt.plot(angle2/pi,((z1*cos(angle)+z2*cos(angle2))).imag,'b--')
		
		plt.figure()
		plt.plot(angle/pi,-1*z1*cos(angle).imag,'r')
		plt.plot(angle/pi,z2*cos(angle2).imag,'b')
		'''
		plt.figure()
		plt.plot(angle,angle2.real/pi,'r')
		plt.plot(angle,angle2.imag/pi,'b')
		plt.plot(angle,-1*abs(angle2.imag)/pi,'g')
		plt.show()
		exit()
	
	return R
	
def Transmission(angle,e1,e2,pol='TE',debug=False):
	'''
	Return complex transmission coefficient for two dielectric interface.  Magnitude is abs(T), phase is np.angle(T)
	
	Sign convention taken from Saleh and Teich Fundamentals of Photonics, because it makes things work in
	DielectricSlab.py
	
	'''
	n1 = sp.emath.sqrt(e1)
	n2 = sp.emath.sqrt(e2)

	angle2 = sp.arcsin(n1/n2*sin(angle))
	R = Reflection(angle,e1,e2,pol)

	if   pol=='TE': T = 1 + R
	elif pol=='TM': T = (1 + R) * cos(angle)/cos(angle2)
	
	return T

def Pairs(list1,list2):
	'''
	returns a list of all possible tuples of form ([list1],[list2])
	
	eg. Pairs([a,b],[1,2]) = [(a,1),(a,2),(b,1),(b,2)]
	'''
	
	L1,L2 = np.meshgrid(list1,list2)
	L1 = L1.flatten()
	L2 = L2.flatten()
	
	return zip(L1,L2)
	
	
def is_number(s):
	'''
	tests if string s is a number
	'''	
	try:
		float(s)
		return True
	except ValueError:
		return False	
		
def getExtent(xs,ys):
	x0 = np.amin(xs)
	y0 = np.amin(ys)
	x1 = np.amax(xs)
	y1 = np.amax(ys)
	
	return [x0,x1,y0,y1]
	
def getClosestIndex(array,target):
	'''
	finds the index of the element in array that is closest numerically to target
	'''
	return np.abs(array-target).argmin()
	
def PlotSeries(xs,yarray,cmap='RdYlBu',ax=None):
	'''
	display an array of data as lines using matplotlib colormaps
	'''

	if ax == None:
		plt.figure()
		ax = plt.subplot(111)

	n = np.shape(yarray)[1]
	norm = mpl.colors.Normalize(vmin=0, vmax=n)
	sm = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)

	print np.shape(ax)

	if np.shape(ax)==():
		for i in range(n):
			ax.plot(xs,yarray[:,i],color=sm.to_rgba(i),lw=2)

	elif np.shape(ax)[0]>1: # "waterfall" stack of axes

		if np.shape(ax)[0] != n:
			print np.shape(ax)[0]
			print n 
			print 'PlotSeries: incorrect number of axes for data'

		else:
			for i in range(n):
				ax[i].plot(xs,yarray[:,i],color=sm.to_rgba(i),lw=2)

	return ax 

def ev2nm(e):
	J = e * q
	nm = h*c/J * 1e9
	return nm 

def nm2ev(wl):
	return h*c/(wl*q) * 1e9

def loadAbajoData(material,units='wl'):
	
	matDict = {
		'al'   :'abajo_aluminum_palik.dat',
		'au'   :'abajo_gold_jc.dat',
		'cu'   :'abajo_copper_jc.dat',
		'co'   :'abajo_cobalt_jc.dat',
		'pt'   :'abajo_platinum.dat',
		'aSi'   :'abajo_aSi_palik.dat',
		'si'   :'abajo_Si_Aspnes1983.dat',
		'al2o3':'abajo_alumina_palik.dat'
	}

	fname = matDict[material]

	data = np.loadtxt(DATA_PATH + '/Materials/' + fname, skiprows=11)
	
	eV = data[:,0]
	er = data[:,1]
	ei = data[:,2]

	wl = ev2nm(eV)

	if units == 'wl':
		# reverse order of elements to be monotonically increasing
		wl = wl[::-1]
		er = er[::-1]
		ei = ei[::-1]

	unitsDict = {
			'wl':wl,
			'ev':eV
		}
		
	return unitsDict[units], er, ei

def loadDatathiefData(fname):

	data = np.loadtxt(DATA_PATH + '/Materials/' + fname, skiprows=1, delimiter=", ")

	# sort data by x-coordinate
	ind = data[:,0].argsort()

	
	x = data[:,0][ind]
	y = data[:,1][ind]
	
	return x,y


def getEps(material,wl):
	
	if material == 'air': return np.ones(len(wl))

	loaderDict = {
		'al':loadAbajoData,
		'au':loadAbajoData,
		'co':loadAbajoData,
		'cu':loadAbajoData,
		'pt':loadAbajoData,
		'al2o3':loadAbajoData,
		'aSi':loadAbajoData,
		'si':loadAbajoData,

		'a-gete':loadEpsDatathief,
		'c-gete':loadEpsDatathief,
		'a-oldgete':loadEpsDatathief,
		'c-oldgete':loadEpsDatathief,
		'a-gst225':loadEpsDatathief,
		'c-gst225':loadEpsDatathief,
		'a-gst326':loadEpsGST326,
		'c-gst326':loadEpsGST326
	}

	wls,er,ei = loaderDict[material](material)

	# print np.amin(wls), np.amax(wls), wl
	# print material
	# print wls
	# print er

	# plot source data for reference
	# plt.plot(wls, er, 'r')
	# plt.plot(wls, ei, 'b')
	# plt.show()

	interp_er = interp1d(wls, er, kind='linear', bounds_error=False, fill_value=np.nan)
	interp_ei = interp1d(wls, ei, kind='linear', bounds_error=False, fill_value=np.nan)

	er2 = interp_er(wl)
	ei2 = interp_ei(wl)

	return er2 + 1j*ei2

def getEpsDrude(m, wl):
    """
    Returns optical properties of ideal Drude metal
    """
    c = 3e8 # m/s
    h = 6.626e-34 # J-s
    q = 1.602e-19 # C
    nm = 1e-9
    
    if m == 'Ag':
        e0 = 5.
        wp = 9.2#9.2159 # [eV]
        gamma = 0.021#0.0212 # [eV]
    elif m == 'Au':
        e0 = 9.
        wp = 9.#9.1
        gamma = 0.066#0.072
    elif m == 'Cu':
    		e0 = 4.5
    		wp = 1.34e16 * h/q
    		gamma = 1/(6.9e-15) * h/q
    else: print 'Invalid metal'
    
    w = (h * c) / (wl*nm * q)
    
    eps_real = e0 - (wp**2) / (w**2 + gamma**2)
    eps_imag = gamma * wp**2 / w / (w**2 + gamma**2)
    eps = eps_real - 1j*eps_imag
        
    return eps

def loadEpsDatathief(material,units='wl'):

	matDict = {
		'a-gst225':['gst225_e1_a.txt','gst225_e2_a.txt'],
		'c-gst225':['gst225_e1_c.txt','gst225_e2_c.txt'],
		'a-gete'  :['gete_e1_a.txt','gete_e2_a.txt'],
		'c-gete'  :['gete_e1_c.txt','gete_e2_c.txt'],
		'a-oldgete'  :['old_gete/GeTe_a_real.txt','old_gete/GeTe_a_imag.txt'],
		'c-oldgete'  :['old_gete/GeTe_c_real.txt','old_gete/GeTe_c_imag.txt']
	}

	fname_r = matDict[material][0]
	fname_i = matDict[material][1]

	xr,yr = loadDatathiefData(fname_r)
	xi,yi = loadDatathiefData(fname_i)

	# create common set of x values
	xmin = np.amin(np.hstack([xr,xi]))
	xmax = np.amax(np.hstack([xr,xi]))
	x = np.linspace(xmin,xmax,1000)

	interp_r = interp1d(xr, yr, kind=1, bounds_error=False, fill_value=np.nan)
	interp_i = interp1d(xi, yi, kind=1, bounds_error=False, fill_value=np.nan)

	er = interp_r(x)	
	ei = interp_i(x)	
	
	eV = x

	wl = ev2nm(eV)

	if units == 'wl':
		# reverse order of elements to be monotonically increasing
		wl = wl[::-1]
		er = er[::-1]
		ei = ei[::-1]

	unitsDict = {
			'wl':wl,
			'ev':eV
		}
		
	return unitsDict[units], er, ei



def example_getEps_GST():

	ev = np.linspace(0.01,8,1000)
	wl = ev2nm(ev)

	# eps = getEps('c-gete',wl)
	eps_225 = getEps('c-gst225',wl)
	eps_326 = getEps('c-gst326',wl)

	n_225 = sp.emath.sqrt(eps_225)
	n_326 = sp.emath.sqrt(eps_326)

	fig, ax = plt.subplots(2,figsize=(7,10),sharex=True)

	ax[0].plot(wl,n_225.real,'r')
	ax[0].plot(wl,n_326.real,'r:')

	ax[1].plot(wl,n_225.imag,'b')
	ax[1].plot(wl,n_326.imag,'b:')

	plt.xlim(1500,6000)

	plt.show()
	
	return 0

def loadEpsGST326(material,units='wl'):
	
	'''
	Data from Ann-Katrin's ellipsometry measurements
	wl - wavelength in nm
	'''

	phase = material[0]
	
	asdep  = np.loadtxt(DATA_PATH + '/Materials/GST326_asdep_585nm_eps_SELFMADE.xy')
	al_cry = np.loadtxt(DATA_PATH + '/Materials/GST326_al_cry_506nm_eps_SELFMADE.xy')
	
	step = 1
	ev_a = asdep[::step,0]
	e1_a = asdep[::step,1]
	e2_a = asdep[::step,2]
	
	ev_c = al_cry[::step,0]
	e1_c = al_cry[::step,1]
	e2_c = al_cry[::step,2]
	
	eV = ev_a if phase=='a' else ev_c
	er = e1_a if phase=='a' else e1_c
	ei = e2_a if phase=='a' else e2_c

	wl = ev2nm(eV)

	if units == 'wl':
		# reverse order of elements to be monotonically increasing
		wl = wl[::-1]
		er = er[::-1]
		ei = ei[::-1]

	unitsDict = {
			'wl':wl,
			'ev':eV
		}
		
	return unitsDict[units], er, ei
		
	return e1 - e2*1j

def stackPoints(arrays):
	'''
	merge a series of unequal length arrays by padding each to a common length with np.nan.

	Upon completion, indexing 'result[0]' will return a series of the 0th index points from each
	of the source arrays.
	'''

	# Find largest length
	N = 0
	for a in arrays:
		if len(a) > N: N = len(a)

	result = np.zeros((N,len(arrays)), dtype=complex)

	# Pad arrays and append to output
	for i,a in enumerate(arrays):
		pad = np.ones(N - len(a)) * complex(np.nan,np.nan)
		result[:,i] = np.append(a,pad)

	return result

def test_stackPoints():

	a = np.array([1,2])
	b = np.array([3,4,5])

	print stackPoints([a,b])
	print stackPoints([a,b])[0]
	print stackPoints([a,b]).shape

def Closest(array,value):
	'''
	return index in array of the element closest magnitude to a target value
	'''
	idx = np.where(np.abs(array-value) == np.amin(np.abs(array-value)))[0][0]

	return idx


if __name__ == '__main__':
	test_stackPoints()
