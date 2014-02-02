#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

def main():
	
	e1=2+0j
	e2=1+0j
	
	angle = np.linspace(0,sp.pi/2,1000)
	R = []
	
	for t in angle:
		Rt = putil.Reflection(t,e1,e2)
		R.append(Rt)
		
	R = np.array(R)
	R = np.abs(R)
	
	plt.plot(angle*180/sp.pi,R)
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

