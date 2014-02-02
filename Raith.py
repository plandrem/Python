#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

RAITHPATH = "/Users/Patrick/Documents/PhD/Raith/ASCII/"

def printAscii(pts,fname=None):
	
	outStr = ''
	for pt in pts:
		outStr += '1 100.0 %u\n' % pt[-1]

		for crd in pt[:-1]:
			outStr += '%.3f %.3f\n' % crd

		outStr += '#\n'

	print outStr

	# output to file

	if not fname: fname = RAITHPATH + 'Raith_Output.txt'

	with open(fname,'w') as f:
		f.write(outStr)



def NanowireArray():
	'''
	all sizes in um.
	'''

	minSize = 50    * 1e-3
	maxSize = 1000  * 1e-3
	step		= 20    * 1e-3
	pitch	  = 10
	length  = 50

	offset = 0 #represents shift from origin to lower left corner of current nanowire

	sizeArray = np.arange(minSize,maxSize+0.1*step,step)
	N = len(sizeArray)

	pts = [] # polygon coordinates for output. Each element is of format [(x,y),(x,y),(x,y),(x,y)]. Order is l. left, l. right, u. right, u. left, l. left (returns to start for formatting purposes).

	for i in range(N):

		ll = (offset,0)
		lr = (offset+sizeArray[i],0)
		ur = (offset+sizeArray[i],length)
		ul = (offset,length)

		layer = 1

		pt = [ll,lr,ur,ul,ll,layer]

		pts.append(pt)

		offset += pitch

	fname = RAITHPATH + 'NW_array.asc'
	printAscii(pts,fname)





	return

if __name__ == '__main__':
  NanowireArray()
