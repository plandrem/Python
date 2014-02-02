#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import sys
import os
import time

from scipy import cos, sin, pi
from scipy.stats import linregress

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
	


def main():
	
	return 0

if __name__ == '__main__':
	main()

