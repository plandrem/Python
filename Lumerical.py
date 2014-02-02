#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os
import csv

def LoadLines(fname):
	'''
	Returns list of 1D line plot data from exported Lumerical txt file
	'''
	
	'''
	with open(fname,'rb') as f:
		r = csv.reader(f)
		for i in range(3):
			r.next()
		
		output = []
		
		l = []
		
		row = r.next()
		if row != []:
			l.append(row)
		else:
			output.append(l)
			for i in range(5):
				r.next()
	'''
	record = False
	
	with open(fname,'rb') as f:
		r = csv.reader(f)
		
		x = []
		y = []
		output = []
		
		for row in r:
			if row != []:
				if putil.is_number(row[0]):
					x.append(float(row[0]))
					y.append(float(row[1]))
					record = True
				else:				
					if record == True:
						x = np.array(x)
						y = np.array(y)
						output.append((x,y))
						x = []
						y = []
						record = False
		x = np.array(x)
		y = np.array(y)
		output.append((x,y))
	return output
	
def PlotLines(fname):
	
	data = LoadLines(fname)
	
	for l in data:
		x = l[0]
		y = l[1]
		plt.plot(x,y)

	plt.show()						

def main():
	
	return 0

if __name__ == '__main__':
	main()

