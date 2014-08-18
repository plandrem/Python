#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos
from scipy.interpolate import interp1d

import numpy.fft as FFT

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

'''
Method from A. Feldman, "Algorithm for solutions of the thermal diffusion equation in a stratefied medium with a modulated heating source",
High Temperatures - High Pressures, 1999.
'''

# Materials Definitions
# k - conductivity [W/m K]
# p - density [kg/m^3]
# c - specific heat [J/kg K]
# Source - engineeringtoolbox.com

materials = {
	'air'     :{'k':0.02,'p':1,'c':1000},
	'aluminum':{'k':300,'p':1000,'c':897},
	'alumina' :{'k':30,'p':961,'c':718},
	'quartz'  :{'k':1.3,'p':2000,'c':700},
  'GST'     :{'k':0.56,'p':6000,'c':216}
}

class Layer:

	def __init__(self,material,L):

		self.material = material
		self.L = L * 1e-9
		self.l = L
		self.k = materials[material]['k'] # * (1e-9) # m/nm
		self.p = materials[material]['p'] # * (1e-9)**3
		self.c = materials[material]['c']

		self.D = self.k/(self.p*self.c)

	def setEigenvalue(self,w):
		self.u = sqrt(-1j*w*self.D)
		self.g = self.u*self.k

class Pulse:

	def __init__(self,p,period,res):

		self.p = p 							# pulse duration
		self.period = period		# pulse period (affects frequency resolution)
		self.res = res 					# timestep (affects maximum frequency)

		# create time points for one cycle
		self.t = np.arange(0,period,res)

		# create pulse shape
		self.amp = (self.t <= p)

		# perform fft
		self.fft = FFT.fftshift(FFT.fft(self.amp))

		n = len(self.t)
		d = self.t[1]-self.t[0]
		self.freqs = FFT.fftshift(FFT.fftfreq(n,d))

		self.n = len(self.freqs) # number of frequencies

	def plotFFT(self):

		plt.plot(self.freqs,np.real(self.fft),'r')
		plt.plot(self.freqs,np.imag(self.fft),'b')
	
	def plotPulse(self):

		plt.figure()
		plt.plot(self.t,self.amp,'r',lw=2)



def getPropagationMatrix(w,layer,d=None,reverse=False):

	u = layer.u
	L = layer.L

	# if a custom distance supplied, override layer property
	if d or d==0: L = d

	# if computing the stack in reverse, negate L
	if reverse: L *= -1

	u11 = exp(u*L)
	u12 = 0
	u21 = 0
	u22 = exp(-u*L)

	U = np.matrix([[u11,u12],[u21,u22]])

	return U

def getBoundaryMatrix(w,la,lb):

	ga = la.g
	gb = lb.g

	g11 = gb + ga
	g21 = gb - ga
	g12 = gb - ga
	g22 = gb + ga

	G = np.matrix([[g11,g12],[g21,g22]]) / (2*gb)

	return G

def getTempDistribution(w,stack,xo,q=1,res=1,debug=False):
	N = len(stack) - 2

	res *= 1e-9
	xo *= 1e-9

	# find location of heat plane (jth layer)
	j,a,b = getCurrentLayer(xo,stack,split=True)

	# initialize list of depths per layer
	xs = [np.arange(0,l.L,res) for l in stack]

	# eliminate redundant x=0 point at semi-infinite layer boundaries
	xs[0] = xs[0][1:]
	xs[-1] = xs[-1][1:]

	# initialize temperature values per layer
	ts_left  = []
	ts_right = []

	'''
	LEFT SIDE
	'''

	# solve stack from the left of heat plane until current layer
	A = np.matrix([[1],[0]])

	for n in range(j):

		if n == 0:
			# get decaying tail in left semi-inf region
			t = [propagateTemp(w,A,d,stack[0]) for d in xs[0][::-1]*-1]
		else:
			la = stack[n-1]
			lb = stack[n]
			U = getPropagationMatrix(w,lb)
			G = getBoundaryMatrix(w,la,lb)

			t = [propagateTemp(w,G*A,d,stack[n]) for d in xs[n]]
			A = U*G*A

		ts_left.append(t)



	# apply final matrices to arrive at target point

	lj   = stack[j]
	lj_1 = stack[j-1]

	Uj = getPropagationMatrix(w,lj,a)
	Gj = getBoundaryMatrix(w,lj_1,lj)

	t = [propagateTemp(w,Gj*A,d,stack[j]) for d in xs[j]]
	t = np.array(t)
	ts_left.append(t)

	A = Uj*Gj*A # Final form of matrix A

	# stitch temperatures for each zone
	ts_left = np.hstack(ts_left)

	# stitch xs
	xs_flat = np.hstack(xs[:j+1])
	xs_left = np.arange(0,len(xs_flat))*res

	'''
	RIGHT SIDE
	'''
	

	# solve stack from the right of heat plane until current layer
	B = np.matrix([[0],[1]])

	for n in range(N+1-j):

		s2 = stack[::-1]
		x2 = xs[::-1]

		if n == 0:
			# get decaying tail in left semi-inf region
			t = [propagateTemp(w,B,d,s2[0]) for d in x2[0]]
		else:
			la = s2[n-1]
			lb = s2[n]
			U = getPropagationMatrix(w,lb,reverse=True)
			G = getBoundaryMatrix(w,la,lb)

			t = [propagateTemp(w,G*B,d,s2[n],reverse=True) for d in x2[n][::-1]]
			B = U*G*B
		
		ts_right.append(t)



	# apply final matrices to arrive at target point

	lj   = stack[j]
	ljp1 = stack[j+1]

	Uj = getPropagationMatrix(w,lj,b,reverse=True)
	Gj = getBoundaryMatrix(w,ljp1,lj)
	
	t = [propagateTemp(w,Gj*B,d,stack[j],reverse=True) for d in xs[j][::-1]]
	t = np.array(t)
	ts_right.append(t)

	B = Uj*Gj*B # Final form of B

	# stitch temperatures for each zone
	ts_right = np.hstack(ts_right[::-1])

	# stitch xs
	xs_flat = np.hstack(xs[j:])
	xs_right = np.arange(0,len(xs_flat))*res

	# scale right and left solutions using heat plane boundary condition
	To = getEndTemperature(stack,xo,A,B,q=q,side='top')
	Tb = getEndTemperature(stack,xo,A,B,q=q,side='bottom')

	ts_left  *= To
	ts_right *= Tb

	# shift right-side x values to absolute position
	for i in range(j):
		xs_right += stack[i].L

	# find index of heat plane for each side
	idxl = np.argmin(abs(xs_left  - (xo + stack[0].L)))
	idxr = np.argmin(abs(xs_right - (xo + stack[0].L)))
	
	tf = np.hstack([ts_left[:idxl],ts_right[idxr:]])
	xf = np.hstack([xs_left[:idxl],xs_right[idxr:]])

	if debug:
		plt.plot(xs_left ,abs(ts_left ),'r:',lw=2)
		plt.plot(xs_right,abs(ts_right),'b:',lw=2)

		# plt.figure()
		plt.plot(xf,np.abs(tf),'k')
		plt.show()

	return xf,tf

def getPrimaryMatrices(w,stack,xo):
	N = len(stack) - 2

	# find location of heat plane (jth layer)
	j,a,b = getCurrentLayer(xo,stack,split=True)

	# solve stack from the left of heat plane until current layer
	A = np.matrix([[1],[0]])

	for n in range(j-1):
		la = stack[n]
		lb = stack[n+1]
		U = getPropagationMatrix(w,lb)
		G = getBoundaryMatrix(w,la,lb)

		A = U*G*A

	# apply final matrices to arrive at target point

	lj   = stack[j]
	lj_1 = stack[j-1]

	Uj = getPropagationMatrix(w,lj,a)
	Gj = getBoundaryMatrix(w,lj_1,lj)
	A = Uj*Gj*A

	# solve stack from the right of heat plane until current layer
	B = np.matrix([[0],[1]])

	for n in range(N-j):
		la = stack[::-1][n]
		lb = stack[::-1][n+1]
		U = getPropagationMatrix(w,lb,reverse=True)
		G = getBoundaryMatrix(w,la,lb)

		B = U*G*B

	# apply final matrices to arrive at target point

	lj   = stack[j]
	ljp1 = stack[j+1]

	Uj = getPropagationMatrix(w,lj,b)
	Gj = getBoundaryMatrix(w,ljp1,lj)
	B = Uj*Gj*B


	return A,B

def getEndTemperature(stack,xo,A,B,q=1,side='top'):

	# find location of heat plane (jth layer)
	j = getCurrentLayer(xo,stack)
	
	gj = stack[j].g

	if side == 'top':      return q/(2*gj) * (B[0,0] + B[1,0])/(A[0,0]*B[1,0] - A[1,0]*B[0,0])
	elif side == 'bottom': return q/(2*gj) * (A[0,0] + A[1,0])/(A[0,0]*B[1,0] - A[1,0]*B[0,0])

def getStackHeight(stack):
	h = 0
	for l in stack[1:-1]:
		h += l.L

	return h

def getCurrentLayer(x,stack,split=False):
	'''
	for a depth x within stack, returns index in stack of the current layer
	if x coincides with a layer boundary, x is defined to be within the left layer (lower index)

	if split, also return the distances to the left and right boundaries from x
	'''

	idx = -1

	h = 0
	for i,l in enumerate(stack[1:]):

		if h >= x:
			idx = i
			a = x - (h-stack[i].L)
			b = h - x
			break
		
		h += l.L

	# if not found, must be in bottom semi-inf region:
	if idx == -1:
		idx = len(stack)-1
		a = x - h
		b = 0

	if split: return idx,a,b
	else: return idx

def test_getCurrentLayer():

	idx,a,b = getCurrentLayer(520,stack,split=True)
	print stack[idx].material
	print a,b

def setEigenvalues(stack,w):
	for l in stack:
		l.setEigenvalue(w)

def propagateTemp(w,T,x,layer,reverse=False):
	'''
	T is the input temperature vector
	x is the distance into the layer
	'''

	U = getPropagationMatrix(w,layer,d=x,reverse=reverse)

	Tx = U*T

	t = Tx[0,0]+Tx[1,0]

	return t
		
def plotTemp(s,x,t,xo,mode='real'):

	x *= 1e9

	fig = plt.figure()

	if mode == 'abs':
		plt.plot(x,abs(t),'r',lw=2)
	elif mode == 'abszeroed':
		plt.plot(x,abs(t)-np.amax(abs(t)),'r',lw=2)
	elif mode == 'all':
		plt.plot(x,np.real(t),'r',lw=2)
		plt.plot(x,np.imag(t),'b',lw=2)
		plt.plot(x,abs(t),'k',lw=2)
	elif mode == 'real':
		plt.plot(x,np.real(t),'r',lw=2)

	z = 0
	for l in s[:-1]:
		z += l.l
		plt.axvline(z,color='k',ls='-')

	# show thermal source
	plt.axvline(xo + s[0].l,color='r',ls=':',lw=3)

	plt.xlim(0,sum([l.l for l in s]))

	return fig

def plotTemps(s,x,ts,xo,mode='real'):

	x *= 1e9

	fig = plt.figure()

	for t in ts:
		if mode == 'abs':
			plt.plot(x,abs(t),'r',lw=2)
		elif mode == 'abszeroed':
			plt.plot(x,abs(t)-np.amax(abs(t)),'r',lw=2)
		elif mode == 'real':
			plt.plot(x,np.real(t),'r',lw=2)
		elif mode == 'all':
			plt.plot(x,np.real(t),'r',lw=2)
			plt.plot(x,np.imag(t),'b',lw=2)
			plt.plot(x,abs(t),'k',lw=2)

	z = 0
	for l in s[:-1]:
		z += l.l
		plt.axvline(z,color='k',ls='-')

	# show thermal source
	plt.axvline(xo + s[0].l,color='r',ls=':',lw=3)

	plt.xlim(0,sum([l.l for l in s]))

	return fig

def plotPhase(s,xo,ws,x=0,q=1,res=1):
	
	phis = []
	for w in ws:
		xs,ts = getTempSinglePlane(s,w,xo,q=1,res=res)

		# interpolate t as location x
		interp_r = interp1d(xs,ts.real,kind='linear')
		interp_i = interp1d(xs,ts.imag,kind='linear')

		tr = interp_r(x)
		ti = interp_i(x)

		t = tr + 1j*ti

		phis.append(np.angle(t,deg=True))

	fig,ax = plt.subplots(1)
	ax.plot(ws,phis,'bo')
	ax.set_ylim(0,90)
	ax.set_xlabel(r'Frequency ($\omega$)')
	ax.set_ylabel('Phase Shift (degrees)')

def getTempSinglePlane(stack,w,xo,q=1,res=1):

	# Set simulation frequency (Hz)
	setEigenvalues(stack,w)

	x,t = getTempDistribution(w,stack,xo,q=q,res=res,debug=False)

	return x,t

def advanceTime(ws,ts,time=1e-9):
	
	newTs = []
	for w,t in zip(ws,ts):

		t2 = t*exp(1j*w*time)
		newTs.append(t2)

	return newTs

def test_advanceTime(s,xo):

	# Choose frequencies
	w = 1
	ws = np.array([w])
	
	ts = []
	for w in ws:
		x,t = getTempSinglePlane(s,w,xo,q=1,res=10)
		ts.append(t)

	T = []

	times = np.linspace(0,2*pi/w,100)
	for time in times:

		ts_new = advanceTime(ws,ts,time)

		T.append(ts_new[0][5])

	plt.figure()
	plt.plot(times,np.real(T),'r')
	plt.plot(times,np.imag(T),'b')




def main():

	p = Pulse(1e-6,10e-6,5e-8)
	p.plotFFT()
	p.plotPulse()
	plt.show()
	exit()

	# Define stack (thicknesses in nm; first and last are semiinfinite):
	air      = Layer('air'     ,1000)
	gst      = Layer('GST'     ,200)
	aluminum = Layer('aluminum',300)
	alumina  = Layer('alumina' ,20)
	quartz   = Layer('quartz'  ,1000)

	# stack = [air,air,quartz,quartz,quartz,air,air]
	stack = [air,quartz,quartz]
	# stack = [air,gst,alumina,aluminum,quartz]

	# Set position of heat plane (0 = top of first finite layer; bottom of stack obtained using getStackHeight(stack) )
	xmax = getStackHeight(stack)
	xo   = 0

	# Choose frequencies
	ws = np.array([1])*1e20
	# ws = np.linspace(1,10,10)*1e6
	# ws = np.logspace(1,5,100)
	
	ts = []
	for w in ws:
		x,t = getTempSinglePlane(stack,w,xo,q=1,res=1)
		ts.append(t)

	ts1 = advanceTime(ws,ts,1e-6)
	ts2 = advanceTime(ws,ts,2e-6)
	ts3 = advanceTime(ws,ts,3e-6)

	T1 = np.sum(ts1,axis=0)
	T2 = np.sum(ts2,axis=0)
	T3 = np.sum(ts3,axis=0)

	# plotTemps(stack,x,[T1,T2,T3],xo,mode='real')
	plotTemp(stack,x,T1,xo,mode='all')


	# plotPhase(stack,xo,ws)
	plt.show()
	return

if __name__ == '__main__':
  main()
