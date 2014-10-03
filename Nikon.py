#!/usr/bin/python

from __future__ import division

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt

import putil
# import Image
import sys

def LoadTIF(fname):
	data = Image.open(fname)
	return data

def getSpectrum(fname):
	data = np.loadtxt(fname,delimiter=',')
	wls = data[:,0]
	counts = data[:,2]
	
	return wls, counts
	
def main():
	
	fs = ['main.txt','bb.txt','dark.txt']
	cs = []
	
	for f in fs:
		wls,counts = getSpectrum(f)
		cs.append(counts)
	
	spectrum = (cs[0]-cs[2])/(cs[1])
	
	#spectrum = putil.SimpleLP(wls,spectrum,0.05)
	
	plt.figure()
	plt.plot(wls,cs[0],'r-')
	plt.plot(wls,cs[1],'b-')
	plt.plot(wls,cs[2],'k-')
	plt.ylim(0,2e5)
	plt.figure()
	plt.plot(wls,spectrum)
	#plt.show()
	
	sname = 'spectrum.csv'
	sdata = np.array((wls,spectrum)).transpose()
	np.savetxt(sname,sdata,delimiter=',')
	
	return 0

def adHoc():

	f = putil.DATA_PATH + '/Rectangular Resonator/Nikon Spectra/2014-09-17 Aaron NWs/processed06_1.txt'

	wls,cts = np.loadtxt(f,delimiter=',').transpose()
	

	plt.figure(figsize=(7,4))

	plt.plot(wls,cts,'getSpectrum')
	plt.xlim(400,800)
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('R')

	plt.tight_layout()

	plt.show()

if __name__ == '__main__':
	adHoc()
