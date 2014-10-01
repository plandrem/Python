#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos, log

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

prop = mpl.font_manager.FontProperties(fname='/Library/Fonts/GillSans.ttc')


def main():

	'''
	Load Data
	'''

	# prefix = '580x200_100nmAl_heat4e16_'
	# prefix = '580x200_100nmAl_20nmOx_20nmCap_heat3e16_'
	# prefix = '580x200_100nmAl_20nmOx_50nmCap_heat3e16_'
	prefix = '580x200_500nmAl_heat3e16_'
	# prefix = '580x200_100nmAl_heat3e16_'
	# prefix = '580x200_100nmAl_20nmOx_20nmCap_heat8e16_pulse20ns_'

	time,bottom = np.loadtxt(putil.DATA_PATH + '/COMSOL/' + prefix + 'bottom.csv', skiprows=8, delimiter=',').transpose()
	time,center = np.loadtxt(putil.DATA_PATH + '/COMSOL/' + prefix + 'center.csv', skiprows=8, delimiter=',').transpose()
	time,top    = np.loadtxt(putil.DATA_PATH + '/COMSOL/' + prefix + 'top.csv'   , skiprows=8, delimiter=',').transpose()

	time /= 1e-9

	'''
	Find Peak points
	'''

	maxTemp = np.amax(top)
	maxidx  = np.where(top==maxTemp)[0][0]
	# maxTime = time[maxidx]
	maxTime = 200

	'''
	Curve fitting
	'''

	decay = lambda x,tau,a: a*exp(-(x-maxTime)/tau) + 273.15
	decay2 = lambda x,tau1,a1,tau2,a2: a1*exp(-(x-maxTime)/tau1) + a2*exp(-(x-maxTime)/tau2) + 273.15

	# popt, pcov = sp.optimize.curve_fit(decay, time[maxidx+1:], top[maxidx+1:], p0=[10,1])
	# tau,a = popt
	# print popt

	popt2, pcov2 = sp.optimize.curve_fit(decay2, time[maxidx+1:], top[maxidx+1:])
	tau1,a1,tau2,a2 = popt2
	print popt2

	'''
	Compute slope at 900K
	'''

	tempCrit = 900

	# find value in top closest to critical temperature
	idxCrit = putil.Closest(top*(time>maxTime),tempCrit)

	dT = top[idxCrit] - top[idxCrit+1]
	dt = -time[idxCrit] + time[idxCrit+1]

	print 'Cooling rate at %u K: %.3f K/ns' % (tempCrit, dT/dt)
	
	# exit()

	'''
	Plot results
	'''

	plt.figure(figsize=(5,4))

	plt.plot(time,top   , color='r', lw=2)
	plt.plot(time,center, color='g', lw=2)
	plt.plot(time,bottom, color='b', lw=2)

	# plt.plot(time[maxidx+1:],decay(time[maxidx+1:],*popt), 'k:')

	plt.plot(time[maxidx+1:],decay2(time[maxidx+1:],*popt2), 'k--')

	plt.axvline(maxTime, color='k', ls=':', lw=0.5)

	plt.xlim(0,500)

	plt.title('Thermal Response', fontproperties=prop, fontsize=18)
	plt.xlabel('Time (ns)')
	plt.ylabel('Temperature (K)')
	plt.legend(('Top','Center','Bottom'), prop=prop)

	plt.tight_layout()
	plt.show()
	


	return

if __name__ == '__main__':
  main()
