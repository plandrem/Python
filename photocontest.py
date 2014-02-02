#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import putil
import sys
import os

import csv

def main():
	
	with open('submissions.csv','rb') as f:
		with open('submissions_output.csv','wb') as fo:
			rdr = csv.reader(f)
			wtr = csv.writer(fo)
			for row in rdr:
				d = row[2]
				#m,d,y = d.split(' ')
				row.extend(d.split(' '))
				wtr.writerow(row)
			
	return 0

if __name__ == '__main__':
	main()

