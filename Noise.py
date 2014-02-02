#!/usr/bin/env python

from __future__ import division

import os
import sys
import csv

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def SpectrumLoader(fname):
	
	spectra = []
	spectrum = []
	legend = []
	with open(fname, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			
			if len(row) != 0:
				if row[0].startswith('// Series:'):
					if legend != []:
						spectra.append(spectrum)
						spectrum = []
					
					legend.append(row[0].split(':')[-1])
					
				elif row[0].startswith('// Units'): pass
				elif len(row) == 2: spectrum.append(row)
		
		spectra.append(spectrum)
		
		print legend
	return spectra, legend

def getx(spectrum):
		return [x[0] for x in spectrum]
		
def gety(spectrum):
		return [x[1] for x in spectrum]

def SpectrumPlotter(spectra, legend=None):
	
	xs = []
	ys = []
	
	for spectrum in spectra:
		#print spectrum
		#print spectrum
		x = getx(spectrum)
		y = gety(spectrum)
		
		xs.append(x)
		ys.append(y)
		
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	for i in range(len(spectra)):
		ax.semilogx(xs[i],ys[i])

	if legend != None: plt.legend(legend, loc='best')
	
	plt.show()

def resave(spectra,legend,path):
	'''
	saves spectra as a csv file suitable for manipulation in excel
	'''
	
	xs = getx(spectra[0])
	ys = map(gety,spectra)
	
	with open(path,'wb') as f:
		writer = csv.writer(f)
		legend.insert(0,'Frequency')
		writer.writerow(legend)
		
		for i,x in enumerate(xs):
			ylist = [y[i] for y in ys]
			ylist.insert(0,x)
			writer.writerow(ylist)
			
def main():
	
	fname = sys.argv[1]
	spectra, legend = SpectrumLoader(fname)
	
	sname = fname.replace('.TXT','.csv')
	resave(spectra,legend,sname)
	SpectrumPlotter(spectra,legend)
	
	return 0

if __name__ == '__main__':
	main()

