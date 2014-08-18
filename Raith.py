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

# x,y,text,dir='h',layer=0,width=0.5,height=9.0):

class Text:

	def __init__(self):
		self.x 			= 0
		self.y 			= 0
		self.text   = ''
		self.dir    = 'h'
		self.layer  = 0
		self.width  = 0.5
		self.height = 9.0
		self.alignment = 8 # 5 for centered, 8 for bottom-left

class Ascii:

	def __init__(self):

		'''
		polygons are of form ([xy-pairs],layer)
		text is a list of Text objects
		'''

		self.poly = []
		self.text = []

	def addGrating(self,x0=0,y0=0,width=500,pitch=3,length=50,grating_width=200):
		'''
		all sizes in um.
		'''

		width *= 1e-3 # convert from nm to um

		offset = x0 #represents shift from origin to lower left corner of current nanowire

		N = int(np.floor(grating_width / pitch))

		for i in range(N):

			ll = (offset,y0)
			lr = (offset+width,y0)
			ur = (offset+width,y0+length)
			ul = (offset,y0+length)

			layer = 1

			self.addPolygon([ll,lr,ur,ul,ll],layer)

			offset += pitch

		# text label
		t = Text()
		t.x = x0
		t.y = y0-10
		t.text = 'width: %.1f nm; period: %.1f um' % (width*1e3, pitch)
		self.addText(t)


	def addDevice(self,x0=0,y0=0,width=1,length=20,contacts=True,startingLayer=0,rot=False,padEdge=200,text=False):
		'''
		all dimensions in um

		layout is vertical; set rot=True to rotate horizontal
		'''
				
		# NW

		if not rot:
			dx = width
			dy = length + padEdge
		else:
			dx = length + padEdge
			dy = width

		ll = (x0,y0)
		lr = (x0+dx,y0)
		ur = (x0+dx,y0+dy)
		ul = (x0,y0+dy)

		layer = startingLayer

		if width > 0: self.addPolygon([ll,lr,ur,ul,ll],layer)

		# GST Contact Pads
		# Pad 1

		xC = x0 + dx/2.
		yC = y0 + dy/2.

		if not rot:
			xPad = xC
			yPad = y0
		else:
			xPad = x0
			yPad = yC

		ll = (xPad-padEdge/2.,yPad-padEdge/2.)
		lr = (xPad+padEdge/2.,yPad-padEdge/2.)
		ur = (xPad+padEdge/2.,yPad+padEdge/2.)
		ul = (xPad-padEdge/2.,yPad+padEdge/2.)

		layer = startingLayer

		self.addPolygon([ll,lr,ur,ul,ll],layer)

		# Pad 2

		if not rot:
			xPad = xC
			yPad = y0 + padEdge + length
		else:
			xPad = x0 + padEdge + length
			yPad = yC

		ll = (xPad-padEdge/2.,yPad-padEdge/2.)
		lr = (xPad+padEdge/2.,yPad-padEdge/2.)
		ur = (xPad+padEdge/2.,yPad+padEdge/2.)
		ul = (xPad-padEdge/2.,yPad+padEdge/2.)

		layer = startingLayer

		self.addPolygon([ll,lr,ur,ul,ll],layer)

		# Metal Contact Pads
		if contacts:
			# Pad 1

			ll = (xC-45,y0+5)
			lr = (xC+45,y0+5)
			ur = (xC+45,y0+95)
			ul = (xC-45,y0+95)

			layer = 1 + startingLayer

			self.addPolygon([ll,lr,ur,ul,ll],layer)

			# Pad 2

			y2 = y0 + 105 + length

			ll = (xC-45,y2)
			lr = (xC+45,y2)
			ur = (xC+45,y2+90)
			ul = (xC-45,y2+90)

			layer = 1 + startingLayer

			self.addPolygon([ll,lr,ur,ul,ll],layer)

		# text label
		if text:
			t = Text()
			t.x = x0
			t.y = y0-10
			t.text = '%u nm x %u um' % (width*1e3,length)
			self.addText(t)

	def addCross(self,x0=0,y0=0,width=100,thickness=25,layer=1):

		'''
		width: total dimension of device
		thickness: transverse dimension of one "arm" of the addCross

		dimensions in um
		'''

		# 'big' and 'small' dimensions for convenience
		b = width/2.
		s = thickness/2.

		# start at lower left corner of left arm, proceed CCW
		self.addPolygon([
			(x0-b,y0-s),
			(x0-s,y0-s),
			(x0-s,y0-b),
			(x0+s,y0-b),
			(x0+s,y0-s),
			(x0+b,y0-s),
			(x0+b,y0+s),
			(x0+s,y0+s),
			(x0+s,y0+b),
			(x0-s,y0+b),
			(x0-s,y0+s),
			(x0-b,y0+s),
			(x0-b,y0-s)
			],layer)

	def addRectangle(self,x0=0,y0=0,width=100,height=25,layer=1):

		'''
		dimensions in um
		'''

		# 'big' and 'small' dimensions for convenience
		w = width/2.
		h = height/2.

		# start at lower left corner of left arm, proceed CCW
		self.addPolygon([
			(x0-w,y0-h),
			(x0+w,y0-h),
			(x0+w,y0+h),
			(x0-w,y0+h),
			(x0-w,y0-h)
			],layer)

	def addF(self,x0=0,y0=0,width=25,height=100,layer=1):

		'''
		dimensions in um
		'''

		# 'big' and 'small' dimensions for convenience
		w = width/2.
		h = height/10.

		# start at lower left corner, proceed CCW
		self.addPolygon([
			(x0-w,y0-5*h),
			(x0  ,y0-5*h),
			(x0  ,y0-h),
			(x0+w,y0-h),
			(x0+w,y0+h),
			(x0  ,y0+h),
			(x0  ,y0+3*h),
			(x0+w,y0+3*h),
			(x0+w,y0+5*h),
			(x0-w,y0+5*h),
			(x0-w,y0-5*h)
			],layer)

	def addBox(self,x0=0,y0=0,innerw=8,innerh=8,outerw=10,outerh=10,layer=1):

		'''
		Adds a hollow rectangle (ie a rectangular path of fixed width)

		x0,y0 = center of pattern
		innerw, innerh: dimensions of internal rectangle
		outerw, outerh: dimensions of external rectangle
		dimensions in um
		'''

		# simplified dimensions for convenience
		ow = outerw/2.
		oh = outerh/2.
		iw = innerw/2.
		ih = innerh/2.
		
		# start at lower left corner of left arm, proceed CCW
		self.addPolygon([
			(x0-ow,y0-oh),
			(x0+ow,y0-oh),
			(x0+ow,y0+oh),
			(x0-ow,y0+oh),
			(x0-ow,y0-ih),
			(x0-iw,y0-ih),
			(x0-iw,y0+ih),
			(x0+iw,y0+ih),
			(x0+iw,y0-ih),
			(x0-ow,y0-ih),
			(x0-ow,y0-oh)
			],layer)

	def addCircle(self,x0=0,y0=0,inRad=50000,res=500,layer=1,crop=1):

		theta = np.linspace(0,2*pi,res)

		pts = []

		for t in theta:
				x = inRad*cos(t) + x0
				y = inRad*sin(t) + y0

				if not (abs(x) > crop*inRad or abs(y) > crop*inRad):
					pts.append((x,y))

		pts.append(pts[0])

		self.addPolygon(pts,layer)

	def tileWafer(self,cellWidth=10.,rad=50*1e3):

		'''
		repeats current layout to fill a 100cm diameter wafer.
		cellWidth gives unit cell edge in mm
		'''

		cellWidth *= 1e3 # convert from um to mm

		xs = np.arange(-5,6) * cellWidth
		ys = np.arange(-5,6) * cellWidth

		w = h = cellWidth

		tiles = [(x,y) for x in xs for y in ys]

		poly = self.poly[:]

		count = 0

		for t in tiles:
			dx,dy = t

			if (abs(dx)+w/2.)**2 + (abs(dy)+h/2.)**2 < rad**2:

				count += 1

				for p in poly:
					pts,layer = p
					npts = []

					for q in pts:
						npts.append((q[0]+dx,q[1]+dy))

					self.addPolygon(npts,layer)

		print 'Total number of chips per wafer:', count

	def addPolygon(self,pts,layer):

		self.poly.append((pts,layer))

	def addText(self,t):

		self.text.append(t)

	def outputPolygonKlayout(self,pts,layer):

		'''
		pts is a list of form [(x0,y0),(x1,y1),...,(xn,yn)]
		pts must begin and end on the same point

		layer is an integer
		'''

		N = len(pts)

		S = ''
		
		S += 'BOUNDARY \n'
		S += 'LAYER ' + str(layer) + '\n'
		S += 'DATATYPE 0\n'
		S += 'XY '

		for pt in pts:
		  S += '%u: %u\n' % (pt[0]*1e3,pt[1]*1e3)

		S += 'ENDEL\n\n'

		return S

	def outputPolygonRaith(self,pts,layer):
	
		outStr = ''
		for pt in pts:
			outStr += '1 100.0 %u\n' % layer

			for crd in pt:
				outStr += '%.3f %.3f\n' % crd

			outStr += '#\n'

		return outStr

	def outputTextRaith(self,t=Text()):

		'''
		dir = 'h' or 'v', horizontal or vertical
		'''

		rot = 0
		if t.dir == 'v': rot = 90 

		w = '%.3f' % t.width
		
		outStr = ''

		outStr += 'T 100.0 ' + str(t.layer) + ' ' + w + '\n'
		outStr += '%.3f %.3f\n' % (t.x,t.y)
		outStr += '%.3f %.3f\n' % (t.height,rot)
		outStr += '%u %u\n' % (2,0)	# xalign, yalign
		outStr += t.text + '\n'
		outStr += '#\n'

		return outStr

	def outputTextKlayout(self,t=Text()):

		'''
		dir = 'h' or 'v', horizontal or vertical
		'''

		rot = 0
		if t.dir == 'v': rot = 90 

		w = '%.3f' % t.width
		
		outStr = 'TEXT\n'

		outStr += 'LAYER ' + str(t.layer) + '\n'
		outStr += 'TEXTTYPE 0\n'
		outStr += 'PRESENTATION %u\n' % (t.alignment)
		outStr += 'STRANS 0\n'
		outStr += 'MAG ' + str(t.height) + '\n'
		outStr += 'XY %u: %u\n' % (t.x*1e3,t.y*1e3)
		outStr += 'STRING ' + t.text + '\n'	
		outStr += 'ENDEL\n'	

		return outStr

	def outputKlayout(self,fname=None):

		outStr = '''HEADER 600 
BGNLIB 3/7/2014 11:14:30 3/7/2014 11:14:30 
LIBNAME LIB
UNITS 0.001 1e-09 

BGNSTR 3/7/2014 11:14:30 3/7/2014 11:14:30 
STRNAME Contacts

'''

		# iterate through all objects and append to output string
		if len(self.poly) > 0:
			for pt,layer in self.poly:
				outStr += self.outputPolygonKlayout(pt,layer)

		if len(self.text) > 0:
			for t in self.text:
				outStr += self.outputTextKlayout(t)

		# close kLayout syntax

		outStr += 'ENDSTR\n'
		outStr += 'ENDLIB\n'

		# output to file

		if not fname: fname = RAITHPATH + 'Raith_Output.txt'

		print fname

		with open(fname,'w') as f:
			f.write(outStr)

	def outputRaith(self,fname=None):

		outStr = ''

		# iterate through all objects and append to output string
		for pt,layer in self.poly:
			outStr += self.outputPolygonRaith(pt,layer)

		for t,layer in self.text:
			outputTextRaith(t,layer)

		# output to file

		if not fname: fname = RAITHPATH + 'Raith_Output.asc'

		print fname

		with open(fname,'w') as f:
			f.write(outStr)

def addAsciiPolygon(S,pts):
	
	outStr = ''
	for pt in pts:
		outStr += '1 100.0 %u\n' % pt[-1]

		for crd in pt[:-1]:
			outStr += '%.3f %.3f\n' % crd

		outStr += '#\n'

	# print outStr

	S += outStr

	return S

def addAsciiText(S,x,y,text,dir='h',layer=0,width=0.5,height=9.0):

	'''
	dir = 'h' or 'v', horizontal or vertical
	'''

	rot = 0
	if dir == 'v': rot = 90 

	w = '%.3f' % width
	
	outStr = ''

	outStr += 'T 100.0 ' + str(layer) + ' ' + w + '\n'
	outStr += '%.3f %.3f\n' % (x,y)
	outStr += '%.3f %.3f\n' % (height,rot)
	outStr += '%u %u\n' % (2,0)	# xalign, yalign
	outStr += text + '\n'
	outStr += '#\n'

	# print outStr

	# add to main string

	S += outStr
	return S

def printAscii(outStr,fname=None):

	# output to file

	if not fname: fname = RAITHPATH + 'Raith_Output.txt'

	with open(fname,'w') as f:
		f.write(outStr)



def NanowireArray():
	'''
	all sizes in um.
	'''
	fname = RAITHPATH + 'NW_array.asc'
	s = ''

	minSize = 50    * 1e-3
	maxSize = 1000  * 1e-3
	step		= 20    * 1e-3
	pitch	  = 30
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

		# text label
		x = offset - 4.5
		y = -10
		text = '%u' % sizeArray[i]
		s = addAsciiText(s,x,y,text,dir='v')

		offset += pitch

	s = addAsciiPolygon(s,pts)

	return s

def Device(x0=0,y0=0,width=500,length=20,contacts=True,startingLayer=0):
	'''
	width in nm, all else in um
	'''
	
	s = ''

	pts = [] # polygon coordinates for output. Each element is of format [(x,y),(x,y),(x,y),(x,y)]. Order is l. left, l. right, u. right, u. left, l. left (returns to start for formatting purposes).

	width *= 1e-3
	
	# NW
	ll = (x0,y0)
	lr = (x0+width,y0)
	ur = (x0+width,y0+length+200)
	ul = (x0,y0+length+200)

	layer = 1 + startingLayer

	pt = [ll,lr,ur,ul,ll,layer]

	pts.append(pt)

	# GST Contact Pads
	# Pad 1

	xC = x0 + width/2.

	ll = (xC-50,y0)
	lr = (xC+50,y0)
	ur = (xC+50,y0+100)
	ul = (xC-50,y0+100)

	layer = 1 + startingLayer

	pt = [ll,lr,ur,ul,ll,layer]

	pts.append(pt)

	# Pad 2

	y2 = y0 + 100 + length

	ll = (xC-50,y2)
	lr = (xC+50,y2)
	ur = (xC+50,y2+100)
	ul = (xC-50,y2+100)

	layer = 1 + startingLayer

	pt = [ll,lr,ur,ul,ll,layer]

	pts.append(pt)

	# Metal Contact Pads

	if contacts:
		# Pad 1

		ll = (xC-45,y0+5)
		lr = (xC+45,y0+5)
		ur = (xC+45,y0+95)
		ul = (xC-45,y0+95)

		layer = 2 + startingLayer

		pt = [ll,lr,ur,ul,ll,layer]

		pts.append(pt)

		# Pad 2

		y2 = y0 + 105 + length

		ll = (xC-45,y2)
		lr = (xC+45,y2)
		ur = (xC+45,y2+90)
		ul = (xC-45,y2+90)

		layer = 2 + startingLayer

		pt = [ll,lr,ur,ul,ll,layer]

		pts.append(pt)


	s = addAsciiPolygon(s,pts)

	# text label
	x = x0
	y = y0-10
	text = '%u nm x %u um' % (width*1e3,length)
	s = addAsciiText(s,x,y,text,dir='v')

	return s


def Grating(x0=0,y0=0,width=500,pitch=3,length=50,grating_width=200):
	'''
	all sizes in um.
	'''

	s = ''

	width *= 1e-3 # convert from nm to um

	offset = x0 #represents shift from origin to lower left corner of current nanowire

	N = int(np.floor(grating_width / pitch))

	pts = [] # polygon coordinates for output. Each element is of format [(x,y),(x,y),(x,y),(x,y)]. Order is l. left, l. right, u. right, u. left, l. left (returns to start for formatting purposes).

	for i in range(N):

		ll = (offset,y0)
		lr = (offset+width,y0)
		ur = (offset+width,y0+length)
		ul = (offset,y0+length)

		layer = 1

		pt = [ll,lr,ur,ul,ll,layer]

		pts.append(pt)

		offset += pitch

	s = addAsciiPolygon(s,pts)

	# text label
	x = x0
	y = y0-10
	text = 'width: %.1f nm; period: %.1f um' % (width*1e3, pitch)
	s = addAsciiText(s,x,y,text,dir='h')

	return s

def DeviceArray():

	length = 10

	fname = RAITHPATH + 'Devices_' + str(length) + '.asc'	
	s = ''

	ws = np.array([400,500,600,700,800])

	xpitch = 200 #um
	ypitch = 300 + length #um
	maxCol = 10

	x = 0
	y = 0

	for w in ws:
		s += Device(x,y,width=w,length=length)
		if x == (maxCol-1)*xpitch:
			x = 0
			y += ypitch
		else: 
			x += xpitch

	print y/ypitch

	printAscii(s,fname)

def GratingArray():

	fname = RAITHPATH + 'Grating.asc'	
	s = ''

	pmin = 1.5
	pmax = 6
	width = 500 #nm

	ps = np.arange(pmin,pmax + .001,0.1)

	xpitch = 300 #um
	ypitch = 150 #um
	maxCol = 10

	s = addAsciiText(s,0,-30,'Gratings - Period: %.1f to %.1f um; width: %u nm' % (pmin,pmax,width))

	x = 0
	y = 0

	for p in ps:
		s += Grating(x,y,width=width,pitch=p)
		if x == (maxCol-1)*xpitch:
			x = 0
			y += ypitch
		else: 
			x += xpitch

	print y/ypitch

	printAscii(s,fname)

def ContactMask():
	'''
	all sizes in um.

	origin is at chip center
	'''

	A = Ascii()
	pts = []

	# --------------------------------------------------------------------
	# Parameters

	layer = 1

	minGap        = 5 			# minimum gap size allowed (used for fine features)

	chip_w 				= 10000		# width of entire chip
	gapFromEdges 	= 2500		# distance from chip edge to device contacts
	contactEdge 	= 300			# width of device contacts
	subEdge 			= 400 		# width of device "substrate" region
	devGap 				= minGap	# width of gap region between device contacts and "substrate"
	numDev				= 6 			# number of devies per side of the chip
	

	# GST layer
	devPadEdge = 200
	patchEdge = 300

	# common ground
	com_w = chip_w - 2*(gapFromEdges+contactEdge+2*devGap+subEdge)
	com_h = com_w
	com_thick = 500

	# Alignment Markers
	textGap 			 = 1500
	alignGap 			 = 1000
	crossWidth 		 = 200
	crossThickness = 5

	# Guide Rails for Wafer Saw
	rail_w = 100
	railGap = 100

	# Control Parameters
	varLength = True 		# if True, reduces NW length by 25% on each side of central pad


	# --------------------------------------------------------------------
	# Output Key Dimensions

	printStr = ''
	printStr += "Major Chip Dimensions (um):\n"
	printStr += "Chip half-width: %.1f\n" %  (chip_w/2.)
	printStr += "Common Pad Half-width: %.1f\n" % (com_w/2.)
	printStr += "Gap b.w. Contacts and Device Substrate: %.1f\n" % devGap
	printStr += "Contact Pad Size: %.1f\n" % contactEdge
	printStr += "Gap b.w. Contacts and Chip Edge: %.1f\n" % gapFromEdges


	# --------------------------------------------------------------------

	# common pad

	ll = (-com_w/2.,-com_h/2.)
	ul = (-com_w/2., com_h/2.)
	lr = ( com_w/2.,-com_h/2.)
	ur = ( com_w/2., com_h/2.)

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer)


	# Device Contacts and Substrates

	pitch = com_w / float(numDev)
	contactGap = pitch - contactEdge
	printStr += "Pitch b.w. Devices: %.3f\n\n" % pitch

	# for side in ['t','b','l']:
	for side in ['t','b','l','r']:

		if varLength:
			subEdge_l = subEdge
			subEdge_t = subEdge*0.75
			subEdge_r = subEdge*0.5
			subEdge_b = subEdge*0.25
		else:
			subEdge_l = subEdge
			subEdge_t = subEdge
			subEdge_r = subEdge
			subEdge_b = subEdge


		if side == 't':

			printStr += '-- Top Contacts --\n'
			x = -com_w/2. + contactGap/2.
			y = com_h/2. + 2*devGap + subEdge_t

			# modification to cursor to place substrates
			dx = 0
			dy = -subEdge_t - devGap

			# height and width of substrate pad
			h = subEdge_t
			w = contactEdge

		elif side == 'b':
			printStr += '-- Bottom Contacts --\n'
			x = -com_w/2. + contactGap/2.
			y = -1*(com_h/2. + 2*devGap + subEdge_b + contactEdge)

			dx = 0
			dy = devGap + contactEdge

			h = subEdge_b
			w = contactEdge

		elif side == 'l':
			printStr += '-- Left Contacts --\n'
			x = -1*(com_w/2. + 2*devGap + subEdge_l + contactEdge)
			y = -com_h/2. + contactGap/2.

			dx = devGap + contactEdge
			dy = 0

			h = contactEdge
			w = subEdge_l

		elif side == 'r':
			printStr += '-- Right Contacts --\n'
			x = com_w/2. + 2*devGap + subEdge_r
			y = -com_h/2. + contactGap/2.

			dx = -devGap - subEdge_r
			dy = 0

			h = contactEdge
			w = subEdge_r

		for i in range(numDev):

			# Contact

			if side == 'b' and i == numDev - 1:
				# Short for Common Terminal

				ll = (x,y)
				ul = (x,y+subEdge_b+2*devGap+contactEdge)
				lr = (x+contactEdge + contactGap/2.,y)
				ur = (x+contactEdge + contactGap/2.,y+subEdge_b+2*devGap+contactEdge)

				pts = [ll,lr,ur,ul,ll]
				A.addPolygon(pts,layer)

			else:

				ll = (x,y)
				ul = (x,y+contactEdge)
				lr = (x+contactEdge,y)
				ur = (x+contactEdge,y+contactEdge)

				pts = [ll,lr,ur,ul,ll]
				A.addPolygon(pts,layer)

				# Add NW and GST contacts 
				xC = x + contactEdge/2.
				yC = y + contactEdge/2.

				# rotate devices on sides
				rot=False
				if side in ['l','r']: rot=True

				# get device length corrected for potential scaling factor
				ls = {
					'l':subEdge_l,
					'r':subEdge_r,
					'b':subEdge_b,
					't':subEdge_t
					}

				devLength = ls[side] + 2*devGap + 2*(contactEdge-devPadEdge)/2.


				# shift devices on top and right
				if side == 't': yC -= devLength + devPadEdge
				if side == 'r': xC -= devLength + devPadEdge

				A.addDevice(x0=xC,y0=yC,width=1,length=devLength,padEdge=devPadEdge,contacts=False,startingLayer=2,rot=rot)

				# Output Contact Center
				printStr += "(%.3f,%.3f)\n" % (ll[0]+contactEdge/2.,ll[1]+contactEdge/2.)


				# Substrate

				ll = (x+dx,y+dy)
				ul = (x+dx,y+dy+h)
				lr = (x+dx+w,y+dy)
				ur = (x+dx+w,y+dy+h)

				pts = [ll,lr,ur,ul,ll]
				A.addPolygon(pts,layer)

			if side in ['t','b']:
				x += pitch
			else:
				y += pitch

		printStr += '\n'

	# Ellipsometry Patch

	ll = (-patchEdge,-patchEdge)
	lr = ( patchEdge,-patchEdge)
	ur = ( patchEdge, patchEdge)
	ul = (-patchEdge, patchEdge)

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer+1)

	# Metal Grounding Layer

	# left
	ll = (-chip_w/2.+railGap  ,-chip_w/2.+railGap)
	lr = (-gapFromEdges-minGap,-chip_w/2.+railGap)
	ur = (-gapFromEdges-minGap, chip_w/2.-railGap)
	ul = (-chip_w/2.+railGap  , chip_w/2.-railGap)

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer)	

	# right
	ll = (chip_w/2.-railGap  , chip_w/2.-railGap)
	lr = (gapFromEdges+minGap, chip_w/2.-railGap)
	ur = (gapFromEdges+minGap,-chip_w/2.+railGap)
	ul = (chip_w/2.-railGap  ,-chip_w/2.+railGap)

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer)	

	# top
	ll = (-gapFromEdges-minGap, gapFromEdges+minGap)
	lr = ( gapFromEdges+minGap, gapFromEdges+minGap)
	ur = ( gapFromEdges+minGap, chip_w/2.-railGap)
	ul = (-gapFromEdges-minGap, chip_w/2.-railGap)

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer)	

	# bottom
	ll = (-gapFromEdges-minGap,-gapFromEdges-minGap)
	lr = ( gapFromEdges+minGap,-gapFromEdges-minGap)
	ur = ( gapFromEdges+minGap,-chip_w/2.+railGap)
	ul = (-gapFromEdges-minGap,-chip_w/2.+railGap)

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer)	



	# Reference Letters
	
	t = Text()
	t.text = 'L'
	t.x = -(chip_w/2. - textGap)
	t.y = 0
	t.height = 1000
	t.width = 100
	t.alignment = 5
	t.layer = 2
	A.addText(t)

	t = Text()
	t.text = 'T'
	t.x = 0
	t.y = (chip_w/2. - textGap)
	t.height = 1000
	t.width = 100
	t.alignment = 5
	t.layer = 2
	A.addText(t)

	t = Text()
	t.text = 'R'
	t.x = (chip_w/2. - textGap)
	t.y = 0
	t.height = 1000
	t.width = 100
	t.alignment = 5
	t.layer = 2
	A.addText(t)

	t = Text()
	t.text = 'B'
	t.x = 0
	t.y = -(chip_w/2. - textGap)
	t.height = 1000
	t.width = 100
	t.alignment = 5
	t.layer = 2
	A.addText(t)

			
	# Alignment Markers - 3 corners define proper orientation

	printStr += 'Alignment Markers:\n\n'

	for m in ['bottom left','top left','top right']:

		if m == 'bottom left':
			x = -chip_w/2. + alignGap + crossWidth/2.
			y = -chip_w/2. + alignGap + crossWidth/2.
		elif m == 'top left':
			x = -chip_w/2. + alignGap + crossWidth/2.
			y = chip_w/2. - alignGap - crossWidth/2.
		elif m == 'top right':
			x = chip_w/2. - alignGap - crossWidth/2.
			y = chip_w/2. - alignGap - crossWidth/2.

		A.addCross(x0=x,y0=y,width=crossWidth,thickness=crossThickness,layer=layer)
		A.addCross(x0=x,y0=y,width=crossWidth,thickness=crossThickness,layer=layer+1)

		printStr += m + ': (%.3f,%.3f)\n' % (x,y)

	# Bounding box for design purposes

	ll = (-chip_w/2.,-chip_w/2.)
	lr = ( chip_w/2.,-chip_w/2.)
	ur = ( chip_w/2., chip_w/2.)
	ul = (-chip_w/2., chip_w/2.)

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer+2)

	'''
	# Guide Rails for wafer saw

	# top
	ll = (-(chip_w/2. - railGap),(chip_w/2.-railGap-rail_w))
	lr = ( (chip_w/2. - railGap),(chip_w/2.-railGap-rail_w))
	ur = ( (chip_w/2. - railGap),(chip_w/2.-railGap))
	ul = (-(chip_w/2. - railGap),(chip_w/2.-railGap))

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer)

	# bottom
	ll = (-(chip_w/2. - railGap),-(chip_w/2.-railGap-rail_w))
	lr = ( (chip_w/2. - railGap),-(chip_w/2.-railGap-rail_w))
	ur = ( (chip_w/2. - railGap),-(chip_w/2.-railGap))
	ul = (-(chip_w/2. - railGap),-(chip_w/2.-railGap))

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer)
	'''


	maskRadius = 45000
	# A.tileWafer(cellWidth=chip_w*1e-3,rad=maskRadius)

	# # Coarse Alignment Markers
	# s = 30000
	# coarseWidth=5000
	# coarseThick=500

	# A.addCross(x0=-s,y0=-s,width=coarseWidth,thickness=coarseThick,layer=layer)
	# A.addCross(x0=-s,y0=-s,width=coarseWidth,thickness=coarseThick,layer=layer+1)

	# A.addCross(x0=-s,y0=s,width=coarseWidth,thickness=coarseThick,layer=layer)
	# A.addCross(x0=-s,y0=s,width=coarseWidth,thickness=coarseThick,layer=layer+1)

	# A.addCross(x0=s,y0=s,width=coarseWidth,thickness=coarseThick,layer=layer)
	# A.addCross(x0=s,y0=s,width=coarseWidth,thickness=coarseThick,layer=layer+1)

	# A.addCircle(layer=layer+2,inRad=maskRadius,crop=0.9)
	# A.addCircle(layer=layer+3,inRad=50000)

	fname = RAITHPATH + 'ContactMask.txt'
	
	A.outputKlayout(fname=fname)

	print printStr

def RectennaChip(A=None, origin=(0,0), nwWidth=0.7, layer=1, mode='contacts'):
	'''
	all sizes in um except origin (in mm)

	mode = ['contacts','devices'] chooses which elements to append to Ascii object
	nwWidth = cross-sectional width of nanowire devices
	'''

	if not A: A = Ascii()
	pts = []

	# --------------------------------------------------------------------
	# Parameters

	minGap        = 5 			# minimum gap size allowed (used for fine features)

	chip_w 				= 10000		# width of entire chip
	gapFromEdges 	= 2500		# distance from chip edge to device contacts
	contactEdge 	= 300			# width of device contacts
	subEdge 			= 400 		# width of device "substrate" region
	devGap 				= minGap	# width of gap region between device contacts and "substrate"
	numDev				= 6 			# number of devies per side of the chip
	

	# GST layer
	devPadEdge = 200
	patchEdge = 300

	# common ground
	com_w = chip_w - 2*(gapFromEdges+contactEdge+2*devGap+subEdge)
	com_h = com_w
	com_thick = 500

	# Alignment Markers
	textGap 			 = 1500
	alignGap 			 = 1000
	crossWidth 		 = 200
	crossThickness = 5

	# Guide Rails for Wafer Saw
	rail_w = 100
	railGap = 100

	# Control Parameters
	varLength = True 		# if True, reduces NW length by 25% on each side of central pad


	# --------------------------------------------------------------------
	# Output Key Dimensions

	printStr = ''
	printStr += "Major Chip Dimensions (um):\n"
	printStr += "Chip half-width: %.1f\n" %  (chip_w/2.)
	printStr += "Common Pad Half-width: %.1f\n" % (com_w/2.)
	printStr += "Gap b.w. Contacts and Device Substrate: %.1f\n" % devGap
	printStr += "Contact Pad Size: %.1f\n" % contactEdge
	printStr += "Gap b.w. Contacts and Chip Edge: %.1f\n" % gapFromEdges


	# --------------------------------------------------------------------

	# Set Origin; convert from mm to um
	X,Y = origin
	X *= 1e3
	Y *= 1e3

	# common pad

	ll = (X-com_w/2.,Y-com_h/2.)
	ul = (X-com_w/2.,Y+com_h/2.)
	lr = (X+com_w/2.,Y-com_h/2.)
	ur = (X+com_w/2.,Y+com_h/2.)

	pts = [ll,lr,ur,ul,ll]
	if mode == 'contacts': A.addPolygon(pts,layer)


	# Device Contacts and Substrates

	pitch = com_w / float(numDev)
	contactGap = pitch - contactEdge
	printStr += "Pitch b.w. Devices: %.3f\n\n" % pitch

	# for side in ['t','b','l']:
	for side in ['t','b','l','r']:

		if varLength:
			subEdge_l = subEdge
			subEdge_t = subEdge*0.75
			subEdge_r = subEdge*0.5
			subEdge_b = subEdge*0.25
		else:
			subEdge_l = subEdge
			subEdge_t = subEdge
			subEdge_r = subEdge
			subEdge_b = subEdge


		if side == 't':

			printStr += '-- Top Contacts --\n'
			x = -com_w/2. + contactGap/2.
			y = com_h/2. + 2*devGap + subEdge_t

			# modification to cursor to place substrates
			dx = 0
			dy = -subEdge_t - devGap

			# height and width of substrate pad
			h = subEdge_t
			w = contactEdge

		elif side == 'b':
			printStr += '-- Bottom Contacts --\n'
			x = -com_w/2. + contactGap/2.
			y = -1*(com_h/2. + 2*devGap + subEdge_b + contactEdge)

			dx = 0
			dy = devGap + contactEdge

			h = subEdge_b
			w = contactEdge

		elif side == 'l':
			printStr += '-- Left Contacts --\n'
			x = -1*(com_w/2. + 2*devGap + subEdge_l + contactEdge)
			y = -com_h/2. + contactGap/2.

			dx = devGap + contactEdge
			dy = 0

			h = contactEdge
			w = subEdge_l

		elif side == 'r':
			printStr += '-- Right Contacts --\n'
			x = com_w/2. + 2*devGap + subEdge_r
			y = -com_h/2. + contactGap/2.

			dx = -devGap - subEdge_r
			dy = 0

			h = contactEdge
			w = subEdge_r

		for i in range(numDev):

			# Contact

			if side == 'b' and i == numDev - 1:
				# Short for Common Terminal

				ll = (X+x,                            Y-chip_w/2.+railGap)
				ul = (X+x,														Y+y+subEdge_b+2*devGap+contactEdge)
				lr = (X+x+contactEdge + contactGap/2.,Y-chip_w/2.+railGap)
				ur = (X+x+contactEdge + contactGap/2.,Y+y+subEdge_b+2*devGap+contactEdge)

				pts = [ll,lr,ur,ul,ll]
				if mode == 'contacts': A.addPolygon(pts,layer)

			else:

				ll = (X+x,						Y+y)
				ul = (X+x,						Y+y+contactEdge)
				lr = (X+x+contactEdge,Y+y)
				ur = (X+x+contactEdge,Y+y+contactEdge)

				pts = [ll,lr,ur,ul,ll]
				if mode == 'contacts': A.addPolygon(pts,layer)

				# Add NW and GST contacts 
				xC = X + x + contactEdge/2.
				yC = Y + y + contactEdge/2.

				# rotate devices on sides
				rot=False
				if side in ['l','r']: rot=True

				# get device length corrected for potential scaling factor
				ls = {
					'l':subEdge_l,
					'r':subEdge_r,
					'b':subEdge_b,
					't':subEdge_t
					}

				devLength = ls[side] + 2*devGap + 2*(contactEdge-devPadEdge)/2.


				# shift devices on top and right
				if side == 't': yC -= devLength + devPadEdge
				if side == 'r': xC -= devLength + devPadEdge

				if mode == 'devices':
					A.addDevice(x0=xC,y0=yC,width=nwWidth,length=devLength,padEdge=devPadEdge,contacts=False,startingLayer=1,rot=rot)

				elif mode == 'topContacts':
					A.addDevice(x0=xC,y0=yC,width=0,length=devLength,padEdge=devPadEdge,contacts=False,startingLayer=1,rot=rot)

				# Output Contact Center
				printStr += "(%.3f,%.3f)\n" % (ll[0]+contactEdge/2.,ll[1]+contactEdge/2.)


				# Substrate

				ll = (X+x+dx,	 Y+y+dy)
				ul = (X+x+dx,	 Y+y+dy+h)
				lr = (X+x+dx+w,Y+y+dy)
				ur = (X+x+dx+w,Y+y+dy+h)

				pts = [ll,lr,ur,ul,ll]
				if mode == 'contacts': A.addPolygon(pts,layer)

			if side in ['t','b']:
				x += pitch
			else:
				y += pitch

		printStr += '\n'

	# Ellipsometry Patch

	patchX = -800

	ll = (X-patchEdge+patchX,Y-patchEdge)
	lr = (X+patchEdge+patchX,Y-patchEdge)
	ur = (X+patchEdge+patchX,Y+patchEdge)
	ul = (X-patchEdge+patchX,Y+patchEdge)

	pts = [ll,lr,ur,ul,ll]
	if mode == 'devices': A.addPolygon(pts,layer)

	# Front Label

	Fw = 1000
	Fh = 1800
	Fx = 0
	if mode == 'devices': A.addF(x0=X+Fx+Fw/2.,y0=Y,width=Fw,height=Fh,layer=layer)

	# Metal Grounding Layer

	if mode == 'contacts':
		A.addBox(
			x0     = X,
			y0     = Y,
			outerw = chip_w - 2*railGap,
			outerh = chip_w - 2*railGap,
			innerw = 2*(gapFromEdges+minGap),
			innerh = 2*(gapFromEdges+minGap)
			)



def testClass():

	'''
	output a square of width w and height h
	'''
	A = Ascii()

	w = 10
	h = 10

	ll = (-w/2.,-h/2.)
	lr = ( w/2.,-h/2.)
	ul = (-w/2., h/2.)
	ur = ( w/2., h/2.)

	pts = [ll,lr,ur,ul,ll]
	layer = 1

	A.addPolygon(pts,layer)
	A.outputKlayout(fname=RAITHPATH + 'testClass.txt')
	
def testDevice():

	A = Ascii()

	A.addDevice()
	A.outputKlayout(fname=RAITHPATH + 'testDevice.txt')

def testGrating():

	A = Ascii()

	A.addGrating()
	A.outputKlayout(fname=RAITHPATH + 'testGrating.txt')

def testCross():

	A = Ascii()

	A.addCross()
	A.outputKlayout(fname=RAITHPATH + 'testCross.txt')

def testBox():

	A = Ascii()

	A.addBox()
	A.outputKlayout(fname=RAITHPATH + 'testBox.txt')

def ContactsForASML(A=None,origin=(0,0),layer=1):
	'''
	all sizes in um.
	'''

	if not A: A = Ascii()
	pts = []

	# --------------------------------------------------------------------
	# Parameters

	minGap        = 5 			# minimum gap size allowed (used for fine features)

	chip_w 				= 10000		# width of entire chip
	gapFromEdges 	= 2500		# distance from chip edge to device contacts
	contactEdge 	= 300			# width of device contacts
	subEdge 			= 400 		# width of device "substrate" region
	devGap 				= minGap	# width of gap region between device contacts and "substrate"
	numDev				= 6 			# number of devies per side of the chip
	

	# GST layer
	devPadEdge = 200
	patchEdge = 300

	# common ground
	com_w = chip_w - 2*(gapFromEdges+contactEdge+2*devGap+subEdge)
	com_h = com_w
	com_thick = 500

	# Alignment Markers
	textGap 			 = 1500
	alignGap 			 = 1000
	crossWidth 		 = 200
	crossThickness = 5

	# Guide Rails for Wafer Saw
	rail_w = 100
	railGap = 100

	# Control Parameters
	varLength = True 		# if True, reduces NW length by 25% on each side of central pad


	# --------------------------------------------------------------------
	# Output Key Dimensions

	printStr = ''
	printStr += "Major Chip Dimensions (um):\n"
	printStr += "Chip half-width: %.1f\n" %  (chip_w/2.)
	printStr += "Common Pad Half-width: %.1f\n" % (com_w/2.)
	printStr += "Gap b.w. Contacts and Device Substrate: %.1f\n" % devGap
	printStr += "Contact Pad Size: %.1f\n" % contactEdge
	printStr += "Gap b.w. Contacts and Chip Edge: %.1f\n" % gapFromEdges


	# --------------------------------------------------------------------

	X,Y = origin

	# common pad

	ll = (X-com_w/2.,Y-com_h/2.)
	ul = (X-com_w/2.,Y+com_h/2.)
	lr = (X+com_w/2.,Y-com_h/2.)
	ur = (X+com_w/2.,Y+com_h/2.)

	pts = [ll,lr,ur,ul,ll]
	A.addPolygon(pts,layer)


	# Device Contacts and Substrates

	pitch = com_w / float(numDev)
	contactGap = pitch - contactEdge
	printStr += "Pitch b.w. Devices: %.3f\n\n" % pitch

	for side in ['t','b','l','r']:

		if varLength:
			subEdge_l = subEdge
			subEdge_t = subEdge*0.75
			subEdge_r = subEdge*0.5
			subEdge_b = subEdge*0.25
		else:
			subEdge_l = subEdge
			subEdge_t = subEdge
			subEdge_r = subEdge
			subEdge_b = subEdge


		if side == 't':

			printStr += '-- Top Contacts --\n'
			x = -com_w/2. + contactGap/2.
			y = com_h/2. + 2*devGap + subEdge_t

			# modification to cursor to place substrates
			dx = 0
			dy = -subEdge_t - devGap

			# height and width of substrate pad
			h = subEdge_t
			w = contactEdge

		elif side == 'b':
			printStr += '-- Bottom Contacts --\n'
			x = -com_w/2. + contactGap/2.
			y = -1*(com_h/2. + 2*devGap + subEdge_b + contactEdge)

			dx = 0
			dy = devGap + contactEdge

			h = subEdge_b
			w = contactEdge

		elif side == 'l':
			printStr += '-- Left Contacts --\n'
			x = -1*(com_w/2. + 2*devGap + subEdge_l + contactEdge)
			y = -com_h/2. + contactGap/2.

			dx = devGap + contactEdge
			dy = 0

			h = contactEdge
			w = subEdge_l

		elif side == 'r':
			printStr += '-- Right Contacts --\n'
			x = com_w/2. + 2*devGap + subEdge_r
			y = -com_h/2. + contactGap/2.

			dx = -devGap - subEdge_r
			dy = 0

			h = contactEdge
			w = subEdge_r

		for i in range(numDev):

			# Contact

			if side == 'b' and i == numDev - 1:
				# Short for Common Terminal

				ll = (X+x,                            Y-chip_w/2.+railGap)
				ul = (X+x,                            Y+y+subEdge_b+2*devGap+contactEdge)
				lr = (X+x+contactEdge + contactGap/2.,Y-chip_w/2.+railGap)
				ur = (X+x+contactEdge + contactGap/2.,Y+y+subEdge_b+2*devGap+contactEdge)

				pts = [ll,lr,ur,ul,ll]
				A.addPolygon(pts,layer)

			else:

				ll = (X+x,            Y+y)
				ul = (X+x,            Y+y+contactEdge)
				lr = (X+x+contactEdge,Y+y)
				ur = (X+x+contactEdge,Y+y+contactEdge)

				pts = [ll,lr,ur,ul,ll]
				A.addPolygon(pts,layer)

				# Output Contact Center
				printStr += "(%.3f,%.3f)\n" % (ll[0]+contactEdge/2.,ll[1]+contactEdge/2.)

				# Substrate

				ll = (X+x+dx,  Y+y+dy)
				ul = (X+x+dx,  Y+y+dy+h)
				lr = (X+x+dx+w,Y+y+dy)
				ur = (X+x+dx+w,Y+y+dy+h)

				pts = [ll,lr,ur,ul,ll]
				A.addPolygon(pts,layer)

			if side in ['t','b']:
				x += pitch
			else:
				y += pitch

		printStr += '\n'

	# Metal Grounding Layer

	A.addBox(
		x0     = X,
		y0     = Y,
		outerw = chip_w - 2*railGap,
		outerh = chip_w - 2*railGap,
		innerw = 2*(gapFromEdges+minGap),
		innerh = 2*(gapFromEdges+minGap)
		)

	print printStr


def ASMLContacts():

	A = Ascii()

	A.addF(layer=5)

	# ContactsForASML(A=A,origin=(0,6.3e3))
	RectennaChip(A=A, origin=(0,6.4), mode='contacts')
	RectennaChip(A=A, origin=(-3.5,-6.55-2.4), mode='topContacts', nwWidth=0.5)
	RectennaChip(A=A, origin=( 3.5,-6.55-2.4), mode='devices', nwWidth=0.6)
	RectennaChip(A=A, origin=(-3.5,-0.4 -2.4), mode='devices', nwWidth=0.7)
	RectennaChip(A=A, origin=( 3.5,-0.4 -2.4), mode='devices', nwWidth=0.8)

	# Image Boundaries
	A.addRectangle(x0=-0,y0=6.4e3,width=9.8e3,height=9.8e3,layer=3)
	A.addRectangle(x0=-3.5e3,y0=(-6.55-2.4)*1e3,width=4.9e3,height=4.7e3,layer=3)
	A.addRectangle(x0= 3.5e3,y0=(-6.55-2.4)*1e3,width=4.9e3,height=4.7e3,layer=3)
	A.addRectangle(x0=-3.5e3,y0=(-0.4 -2.4)*1e3,width=4.9e3,height=4.7e3,layer=3)
	A.addRectangle(x0= 3.5e3,y0=(-0.4 -2.4)*1e3,width=4.9e3,height=4.7e3,layer=3)

	# Psuedo-boundary for Reticle
	A.addRectangle(width=12e3,height=22.6e3,layer=4)
	A.addRectangle(width=18e3,height=18e3,layer=4)

	fname = RAITHPATH + 'ASMLmask.txt'
	
	A.outputKlayout(fname=fname)


if __name__ == '__main__':

	ASMLContacts()
	# ContactMask()
	# testClass()
	# testCross()
	# testBox()

	# DeviceArray()	

  # NanowireArray()
  # Grating(0,0)
