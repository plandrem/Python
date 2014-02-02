#!/usr/bin/env python

from __future__ import division

import time
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from scipy.special import jn, kn, jv, kv, jvp, kvp
#from scipy.optimize import minimize

import putil
import sys
import os

from scipy import pi,cos,sin,exp,sqrt
csqrt=sp.emath.sqrt

# Define Constants
#e0 = 8.85e-12
#mu = 4*pi*1e-7
#Zo = (mu/e0)**0.5

e0 = 1.
mu = 1.
Zo = 1.
wl = 1.

# Define structure (Rectangular cross section of width a, height b, relative permittivity er)
nr = 1.5
er = nr**2
e1 = er*e0
#nr = er**0.5

# Free space wavelength and corresponding wavevectors
ko = 2*pi/wl
k1 = ko*nr


BB = 1.2 #normalized width
AR = 2. #Aspect Ratio (a/b)
b = wl*BB/(2.*csqrt(nr**2-1))
a = AR*b

'''
a = 2
b = 1
'''

def test_bessel():
	
	x=np.linspace(0,20,100)
	
	for n in range(2):
		plt.plot(x,jn(n,x),'r-')
		plt.plot(x,jvp(n,x),'b-')
	
	#plt.ylim(0,3.5)
	plt.show()

def Kz(P):
	return csqrt(P**2*(nr**2-1)+1)*ko
	
def P2(kz):
	return ((kz/ko)**2-1)/(nr**2-1)


	
def detQ(P,sym=1,N=5,harm=1):
	
	'''
	Matches amplitudes of cylindrical harmonics at boundary of rectangular dielectric region
	
	Inputs:
	P - normalized propagation constant (range 0-1)
	sym - Consider even (0) or odd (1) symmetry across x-ax2is
	N - number of harmonics to consider
	harm - {0,1,'both') determines if even (0), odd (1) or both harmonics are considered in the cylindrical harmonic expansion
	'''
 
	kz = Kz(P)
	h = sqrt(k1**2-kz**2)
	p = sqrt(kz**2-ko**2)
		
	if sym == 1:   # odd  mode w.r.t x-axis
		phi = 0
		
	elif sym == 0: # even mode w.r.t x-axis
		phi = pi/2.
	
	# Instantize matrices
	
	eLA = np.zeros((N,N),dtype=float) # N: number of harmonics we are using
	eLC = np.zeros((N,N),dtype=float)
	hLB = np.zeros((N,N),dtype=float)
	hLD = np.zeros((N,N),dtype=float)
	eTA = np.zeros((N,N),dtype=float)
	eTB = np.zeros((N,N),dtype=float)
	eTC = np.zeros((N,N),dtype=float)
	eTD = np.zeros((N,N),dtype=float)
	hTA = np.zeros((N,N),dtype=float)
	hTB = np.zeros((N,N),dtype=float)
	hTC = np.zeros((N,N),dtype=float)
	hTD = np.zeros((N,N),dtype=float)
	
	for ni in range(N): # arrat (0 to N-1)
		
		# angles used for boundary matching fields at boundary depend on whether current harmonic is odd/even 
		
		# use exclusively even or odd harmonics
		if harm==1: n=2*ni+1
		elif harm==0: n=2*ni
		else: n=ni
		
		j = np.arange(N) # j is an array with size N, having 0 - (N-1)
		
		if n%2==1:     # for odd  harmonic cases (section 2.2) 
			m = j + 1  # m is 1 to N
			theta = (m-0.5)*pi/(2*N)  # theta_m
			
		elif n%2==0:   # for even harmonic cases (section 2.2) 
			if a/b == 1.:
				if sym==0:
					m = j + 1
					theta = (m-0.5)*pi/(2*N)
				else:
					m = j[:-1] + 1
					theta = (m-0.5)*pi/(2*(N-1))
			else:	
				m = j + 1
				theta = (m-0.5)*pi/(2*N)
				
				if sym==1: n += 0
								
		
		# Formulate Matrix Elements
		
		tc = sp.arctan(b/a)

		R = sin(theta)         * (theta < tc) + cos(theta + pi/4.)  * (theta==tc) + -1*cos(theta)     * (theta > tc)
		T = cos(theta)         * (theta < tc) + cos(theta - pi/4.)  * (theta==tc) +    sin(theta)     * (theta > tc)
		rm = a/(2.*cos(theta)) * (theta < tc) + (a**2+b**2)**0.5/2. * (theta==tc) + b/(2.*sin(theta)) * (theta > tc)
		
		'''
		print a,b
		print n
		print theta/pi
		print theta-tc
		print R
		print T
		print
		print sin(theta)
		print cos(theta)
		print
		print rm
		exit()
		'''
		'''
		RR = rm*ko*csqrt(nr**2-1)
		hr = RR*csqrt(1-P**2)
		pr = P*RR
		'''
				
		S = sin(n*theta + phi)
		C = cos(n*theta + phi)
		
		J = jn(n,h*rm)
		Jp = jvp(n,h*rm)
		JJ = n*J/(h**2*rm)
		JJp = Jp/(h)

		K = kn(n,p*rm)
		Kp = kvp(n,p*rm)
		KK = n*K/(p**2*rm)
		KKp = Kp/(p)
		
		# scaling to prevent overflow/underflow
		scaling = True
		
		if scaling:
			d = (a+b)/2.
			Jmult = h**2*d/abs(jn(n,h*b))
			Kmult = p**2*d/abs(kn(n,p*b))
		else:
			Jmult = Kmult = 1.		
		
		
		eLA[:,ni] = J*S                                          * Jmult
		eLC[:,ni] = K*S                                          * Kmult
		hLB[:,ni] = J*C                                          * Jmult
		hLD[:,ni] = K*C                                          * Kmult
		eTA[:,ni] = -1*kz*(JJp*S*R + JJ*C*T)                     * Jmult
		eTB[:,ni] = ko*Zo*(JJ*S*R + JJp*C*T)                     * Jmult
		eTC[:,ni] = kz*(KKp*S*R + KK*C*T)                        * Kmult
		eTD[:,ni] = -1*ko*Zo*(KK*S*R + KKp*C*T)                  * Kmult
		hTA[:,ni] = er*ko*(JJ*C*R - JJp*S*T)/Zo                  * Jmult                 # typo in paper - entered as JJp rather than Jp
		hTB[:,ni] = -kz*(JJp*C*R - JJ*S*T)                       * Jmult
		hTC[:,ni] = -ko*(KK*C*R - KKp*S*T)/Zo                    * Kmult
		hTD[:,ni] = kz*(KKp*C*R - KK*S*T)                        * Kmult
		'''
		
		for mi in m-1:
			eLA[mi,ni] = 'eLA ' + str(mi) + ',' + str(ni)
			eLC[mi,ni] = 'eLC ' + str(mi) + ',' + str(ni)
			hLB[mi,ni] = 'hLB ' + str(mi) + ',' + str(ni)
			hLD[mi,ni] = 'hLD ' + str(mi) + ',' + str(ni)
			eTA[mi,ni] = 'eTA ' + str(mi) + ',' + str(ni)
			eTB[mi,ni] = 'eTB ' + str(mi) + ',' + str(ni)
			eTC[mi,ni] = 'eTC ' + str(mi) + ',' + str(ni)
			eTD[mi,ni] = 'eTD ' + str(mi) + ',' + str(ni)
			hTA[mi,ni] = 'hTA ' + str(mi) + ',' + str(ni)
			hTB[mi,ni] = 'hTB ' + str(mi) + ',' + str(ni)
			hTC[mi,ni] = 'hTC ' + str(mi) + ',' + str(ni)
			hTD[mi,ni] = 'hTD ' + str(mi) + ',' + str(ni)
	'''
	O = np.zeros(np.shape(eLA))	
	
	Q1 = np.hstack((eLA,O,-1*eLC,O))
	Q2 = np.hstack((O,hLB,O,-1*hLD))
	Q3 = np.hstack((eTA,eTB,-1*eTC,-1*eTD))
	Q4 = np.hstack((hTA,hTB,-1*hTC,-1*hTD))
	'''
	Q1 = np.hstack((eLA,O,eLC,O))
	Q2 = np.hstack((O,hLB,O,hLD))
	Q3 = np.hstack((eTA,eTB,eTC,eTD))
	Q4 = np.hstack((hTA,hTB,hTC,hTD))
	'''

	Q = np.vstack((Q1,Q2,Q3,Q4))
		
	#print Q
	#print np.linalg.det(Q)
	#exit()
	
		
	return np.linalg.det(Q)
	
def BothSym(P,N=5):
	'''
	Apply detQ with both symmetries and combine the results
	'''
	
	dqe = detQ(P,sym=0,N=N)
	dqo = detQ(P,sym=1,N=N)
	
	return dqe*dqo
	

def main():
	
	N=10
	p2 = np.linspace(0.01,0.99,100)
	P = sqrt(p2)
	
	#kz = np.linspace(ko,k1,100)
	#p2 = P2(kz)
		
	dqe0 = np.array(putil.maplist(detQ,P,0,N,0))
	dqe1 = np.array(putil.maplist(detQ,P,0,N,1))
	dqo0 = np.array(putil.maplist(detQ,P,1,N,0))
	dqo1 = np.array(putil.maplist(detQ,P,1,N,1))
	
	dqe0 /= np.amax(dqe0)
	dqe1 /= np.amax(dqe1)
	dqo0 /= np.amax(dqo0)
	dqo1 /= np.amax(dqo1)
	
	#for d in [dqe0,dqe1,dqo0,dqo1]:
	#	d = d/np.amax(d)
	
	fig, ax = plt.subplots(2,2,sharex=True)
	
	ax[0,0].plot(p2,dqe0,'r-')
	ax[0,1].plot(p2,dqe1,'b-')
	ax[1,0].plot(p2,dqo0,'c-')
	ax[1,1].plot(p2,dqo1,'m-')
	
	fig2, ax2 = plt.subplots(2,2,sharex=True)

	ax2[0,0].plot(p2,np.sign(dqe0),'r-')
	ax2[0,1].plot(p2,np.sign(dqe1),'b-')
	ax2[1,0].plot(p2,np.sign(dqo0),'c-')
	ax2[1,1].plot(p2,np.sign(dqo1),'m-')
	
	ax[0,0].set_title('sym = even; harm = even')
	ax[0,1].set_title('sym = even; harm = odd')
	ax[1,0].set_title('sym = odd; harm = even')
	ax[1,1].set_title('sym = odd; harm = odd')

	ax2[0,0].set_title('sym x; sym y')
	ax2[0,1].set_title('sym x; anti y')
	ax2[1,0].set_title('anti x; anti y')
	ax2[1,1].set_title('anti x; sym y')
			
	plt.show()
 
def test_detQ():
    N  = 2
    p2 = 0.5
    sym = 1.
    harm = 1.
    
    dq = detQ(sqrt(p2),sym,N,harm)
    print dq
    
	
	
if __name__ == '__main__':
	main()
	#test_detQ()
	#test_bessel()
