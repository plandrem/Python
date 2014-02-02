#!/usr/bin/env python

from __future__ import division

import sys
import os
import csv

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil

debug = False

def LoadCSV(fname):
	
	time = []
	volts = []
	with open(fname, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			time.append(row[3])
			volts.append(row[4])
			
	if debug:
		plt.plot(time,volts)
		plt.show()
		exit()
	
	time = np.array(time,dtype=float)
	volts = np.array(volts,dtype=float)
	return time, volts
	
def LoadDir(path):
	
	times = []
	volts = []
	
	for f in os.listdir(path):
		if os.path.splitext(f)[-1] == '.csv':
			t,v = LoadCSV(os.path.join(path,f))
			times.append(t)
			volts.append(v)
					
	return times,volts
	
def RemoveDC(volts):
	#Subtract constant offset (assumes signal is supposed to be zero at the edges of the waveform)
	
	dc = (volts[0] + volts[-1])/2
	volts -= dc
	
	return volts

def main():
	
	p = sys.argv[1]
	ts, vs = LoadDir(p)
	
	t = ts[0]
	v = vs[0]
	
	putil.GaussianFit(t,v)
	
	return 0

if __name__ == '__main__':
	main()

