#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

atan = sp.arctan
log = sp.log
	
def main():
	
	pi = sp.pi
	
	'''
	#a-Si II
	A = 122
	Eo = 3.45
	Eg = 1.2
	C = 2.54
	eo = 1.15
	'''
	
	A = 1
	Eo = 3
	Eg = 1
	C = 1
	eo = 0

	E = np.linspace(0.5,6,1000)
	
	e2 = A*Eo*C*(E-Eg)**2 / (E*((E**2-Eo**2)**2+C**2*E**2))
	e2 = e2 * (E > Eg)
	
	a = sp.sqrt(4*Eo**2-C**2)
	g = sp.sqrt(Eo**2-(C**2)/2.)
	aln = (Eg**2-Eo**2)*E**2 + Eg**2*C**2 - Eo**2*(Eo**2 + 3*Eg**2)
	aatan = (E**2-Eo**2)*(Eg**2+Eo**2) + Eg**2*C**2
	z4 = (E**2-g**2)**2 + (a**2*C**2)/4.
	
	e1 = eo + 1/2.*A/pi*C/z4*aln/(a*Eo)*log((Eo**2+Eg**2+a*Eg)/(Eo**2+Eg**2-a*Eg)) \
		-A/(pi*z4)*aatan/Eo*(pi-atan((2*Eg+a)/C)+atan((-2*Eg+a)/C)) \
		+2*A*Eo*C/(pi*z4)*(Eg*(E**2-g**2)*(pi+2*atan((g**2-Eg**2)/(a*C)))) \
		-2*A*Eo*C/(pi*z4)*(E**2+Eg**2)/E*log(abs(E-Eg)/(E+Eg)) \
		+2*A*Eo*C/(pi*z4)*Eg*log(abs(E-Eg)*(E+Eg)/sp.sqrt((Eo**2-Eg**2)**2 + Eg**2*C**2))
	
	n = sp.sqrt(e1+1j*e2)
	
	plt.plot(E,e2)
	#plt.plot(E,e1)
	#plt.plot(E,n.real,'r-')
	#plt.plot(E,n.imag,'b-')
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

