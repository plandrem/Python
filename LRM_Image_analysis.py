#!/usr/bin/python
from __future__ import division

import csv
import Image
import pyfits
import pylab as pl
import numpy as np
import matplotlib.image as pltImg
import sys, os
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates
import Tiff

from matplotlib.patches import Wedge

import materials


##############################################################################################################
'''
GLOBAL PARAMETERS
(Includes calibration data for real/BFP distance calculation)
'''


CALPATH = '/home/plandrem/Dropbox/Python/Leakage Radiation Microscopy/calibration.csv'
filetype = '.fit'

# Calibration by diffraction grating.  sin(theta) = m*wl/d.  Find distance to first order maximum and use ratio to convert from arbitrary length to sin(angle)
# DO NOT CHANGE unless recalibrating the analysis

index=1.5																# index of immersion objective (should be ~1.4-1.5)
cal_wl = 535/index														# wavelength at which calibration was performed in nm
cal_spacing = 2500														# periodicity of grating in nm
firstmax = 19.2															# distance from 0th order to 1st order maxima with no calibration (ie. bfp_scaling_factor = 1)
bfp_scaling_factor = cal_wl/cal_spacing/firstmax

# Real image calibration -- distance measured by known periodicity of diffraction gratings.
period = 1.5e-6															# periodicity of grating in m
raw_length = 13.09														# periodicity of grating in image when scaling_factor = 1
real_scaling_factor = period/raw_length

##############################################################################################################




##############################################################################################################
'''
HELPER FUNCTIONS
'''

def SPP_Angle(eps1,eps2,eps3,units='deg'):
	'''
	Calculates leakage radiation angle into medium 3 due to a SPP traveling on the medium 1/2 interface.
	Theta is taken as angle from surface (ie 90 = normal to surface).  Medium 2 should be gold, and eps1 < eps3.
	'''
	
	n3 = emath.sqrt(eps3)
	n3 = n3.real - abs(n3.imag) * 1j									# enforce n' - jn'' convention (sqrt occasionally uses +)
	
	kspp = sqrt(eps1 * eps2 / (eps1 + eps2))
	kspp = kspp.real - abs(kspp.imag) * 1j
	
	if units=='deg': return arccos(1/n3 * kspp) * 180/pi
	if units=='rad': return arccos(1/n3 * kspp)
	
def SimpleLP(xs,ys,freq, plotfft=False):
	'''
	LP filter. Returns original data set (ys) with all frequency components above freq removed.
	'''
	n = len(xs)
	dx = xs[1]-xs[0]
	
	fft_ys = fft(ys)
	freqs = fftfreq(n,dx)
	
	mask = abs(freqs) < freq
	
	fft_filtered = mask * fft_ys
	
	ys_filtered = ifft(fft_filtered)
	
	if plotfft:
		figure()
		plot(freqs,fft_ys,'r-',freqs,fft_filtered,'b-')
	
	return ys_filtered

def GetSPPbounds(wl, width=10):
	'''
	Calculate the inner and outer radius corresponding to SPP leakage.
	Uses BFP distances determined from calibration.
	'''
	
	eps_air = 1
	#eps_metal = materials.getEps('drude-Au', wl)
	eps_metal = materials.getEpsJC('Au', wl)
	eps_sio2 = 2.25
	
	SPPangle = pl.pi/2. - SPP_Angle(eps_air, eps_metal, eps_sio2, units='rad')
	SPPsin = pl.sin(SPPangle)
	SPPsin = SPPsin / bfp_scaling_factor									# Convert from units of sin(theta) into image coordinates
	
	outrad = SPPsin + width
	inrad = SPPsin - 10
	
	radii=(outrad,inrad)
	return radii
	
def GetLinePoints(n,x0,x1,y0,y1):
	"""
	calculates n evenly divided points along the line segment beginning at (x0,y0) and ending at (x1,y1)
	"""
	
	xs = pl.linspace(x0,x1,n)
	ys = pl.linspace(y0,y1,n)
	
	return xs,ys
	
def GaussianFit(xs,ys):

	x = sum(xs*ys)/sum(ys)
	width = pl.sqrt(abs(sum((xs-x)**2*ys)/sum(ys)))						#Standard Deviation

	max = ys.max()

	fit = lambda t : max*pl.exp(-(t-x)**2/(2*width**2))
	
	return fit(xs), width


##############################################################################################################




##############################################################################################################
'''
INTERACTIVE PLOT OBJECTS
'''

class CrossSectionPlot:
	def __init__(self, line, image, xsection):
		self.line = line
		self.image = image
		self.xsec = xsection
		self.xs = list(line.get_xdata())
		self.ys = list(line.get_ydata())
		self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
		self.mode = raw_input('Select Mode ("r" - real, "b" - back focal plane):')
	
		# Reset axes limits for data plot
		xmin = 0
		xmax = self.image.shape[1]
		ymin = 0
		ymax = self.image.shape[0]
		dataplot_bounds = [xmin,xmax,ymin,ymax]
		self.line.axes.axis(dataplot_bounds)
	
		#self.xsection_fig = pl.figure()
		#self.xsection_axes = self.xsection_fig.add_subplot(111)
		#self.xsection, = self.xsection_axes.plot([],[])
		
		# interpolate image
		#self.image_int = interp2d(pl.arange(len(image[0,:])),pl.arange(len(image[:,0])),image,kind='cubic')
		
	### Mouse Click Event Handler ###
	def __call__(self, event):
		
		print "click: x=%f, y=%f" % (event.xdata,event.ydata)
		
		if event.inaxes!=self.line.axes: return
		
		#if len(self.xs) == 2:
		#	self.xs = []
		#	self.ys = []
			
		#self.xs.append(event.xdata)
		#self.ys.append(event.ydata)
		
		#event.button 1=left, 2=wheel, 3=right
		if event.button==1:
			self.xs[0] = event.xdata							
			self.ys[0] = event.ydata

		if event.button==3:
			self.xs[1] = event.xdata							
			self.ys[1] = event.ydata

		self.line.set_data(self.xs, self.ys)
		self.line.figure.canvas.draw()
		
		# Plot cross section of image data
		if len(self.xs) == 2:
			
			# Generate coordinates at which to find image values
			n = 1000
			
			x0 = self.xs[0]
			x1 = self.xs[1]
			y0 = self.ys[0]
			y1 = self.ys[1]
			
			L = pl.sqrt((x1-x0)**2 + (y1-y0)**2)
			print "length: %0.2f" % L
			print "angle: %0.2f" % (L * bfp_scaling_factor/2)
			
			
			xs,ys = GetLinePoints(n,x0,x1,y0,y1)
			
			# Determine coordinate pairs to display in cross section plot
			if self.mode == 'b': L = L * bfp_scaling_factor
			if self.mode == 'r': L = L * real_scaling_factor
			
			xsec_xs = pl.linspace(-L/2.,L/2,n)
			xsec_ys = map_coordinates(self.image,[ys,xs])
			
			# Set new cross section plot axes limits
			xmin = -L/2. #pl.amin(xsec_xs)
			xmax = L/2. #pl.amax(xsec_xs)
			ymin = 0
			ymax = pl.amax(self.image)
			xsec_bounds = [xmin,xmax,ymin,ymax]
			
			# Redraw cross section plot

			self.xsec.set_data(xsec_xs,xsec_ys)
			self.xsec.axes.axis(xsec_bounds)
			self.xsec.figure.canvas.draw()
			
class InteractivePlot:
	
	'''
	Produces 3 figures for manipulation of LRM images:
	
	image_fig, imaxes -- figure and axes objects associated with LRM output image (TIF, fit format).  The image plot within 
							imaxes is named image, while the array used for generating the plot is imdata.
							
	xsection -- figure containing cross section along a line through the diameter of the integration ring.  Points of the line are chosen by 
				shift + clicking (also resizes/repositions circle).  Shift + clicking within this figure will draw an additional
				line for indicating the background noise level.
				
	result_fig -- figure displaying cross section after subtraction of background noise.
	'''
	def __init__(self, fname, beampwr, wls, inttime, radii=None):

		self.image_fig = pl.figure(1)
		self.imaxes = self.image_fig.add_subplot(111)
		
		self.xsection = pl.figure(2)
		self.xsec_axes = self.xsection.add_subplot(111)
		self.xsec_line = self.xsec_axes.plot([],[])[0]
		self.xsec_scatrlineL = self.xsec_axes.plot([],[], color='g')[0]
		self.xsec_scatrlineR = self.xsec_axes.plot([],[], color='g')[0]
		
		self.result_fig = pl.figure(3)
		self.result_axes = self.result_fig.add_subplot(111)
		self.result_line = self.result_axes.plot([],[])[0]
		
		self.center = (0,0)
		self.inrad = 0
		self.outrad = 0
		self.rad1 = 0
		self.rad2 = 0
		
		self.scatr_y0 = 0
		self.scatr_y1 = 0

		self.inttime = inttime
		self.beampwr = beampwr
		self.radii = radii
		self.wls = wls
		self.wli = 0
		self.path = os.path.dirname(fname)

		self.update_image()
		self.line = self.imaxes.plot([],[],lw='3')[0]
				
		#reset image bounds
		xmin = 0
		xmax = pl.shape(self.imdata)[1]
		ymin = 0
		ymax = pl.shape(self.imdata)[0]
		im_bounds = [xmin,xmax,ymin,ymax]
		self.imaxes.axis(im_bounds)
		
		self.pwr = []
		self.pce = []
		
		if radii != None:
			self.inrad = min(radii)
			self.outrad = max(radii)		
		
		self.movering = False											# boolean indicating whether mouse clicks should redraw the ring shape
		self.clickevent = self.image_fig.canvas.mpl_connect('button_press_event', self.click)
		self.keydownevent = self.image_fig.canvas.mpl_connect('key_press_event', self.keydown)
		self.keyupevent = self.image_fig.canvas.mpl_connect('key_release_event', self.keyup)
		
		self.xsec_clickevent = self.xsection.canvas.mpl_connect('button_press_event',self.xsec_click)
		self.xsec_keydownevent = self.xsection.canvas.mpl_connect('key_press_event', self.xsec_keydown)
	
	def update_image(self):

		if self.wli > len(self.wls) - 1:
			pl.close('all')
			return

		wl = int(self.wls[self.wli])
		print wl
		self.imdata = LoadPowerImage(self.path + str(wl) + filetype,wl,self.inttime[self.wli])[0]

		try:
			self.image.set_array(self.imdata)
			pl.figure(1)
			pl.clim(0,pl.amax(self.imdata))
			self.imaxes.figure.canvas.draw()
		except:
			self.image = self.imaxes.imshow(self.imdata, cmap='hot',vmax=None)

		
		self.wli += 1
		
	def update_xsec(self):
		
		# Generate coordinates at which to find image values
		n = 1000
		
		x0 = self.line.get_data()[0][0]
		x1 = self.line.get_data()[0][1]
		y0 = self.line.get_data()[1][0]
		y1 = self.line.get_data()[1][1]
		
		L = pl.sqrt((x1-x0)**2 + (y1-y0)**2)
		print "length: %0.2f" % L
		print "angle: %0.2f" % (L * bfp_scaling_factor/2)
		
		
		xs,ys = GetLinePoints(n,x0,x1,y0,y1)
		
		L = L * bfp_scaling_factor
		
		xsec_xs = pl.linspace(-L/2.,L/2,n)
		xsec_ys = map_coordinates(self.imdata,[ys,xs])
		
		# Set new cross section plot axes limits
		xmin = -L/2. #pl.amin(xsec_xs)
		xmax = L/2. #pl.amax(xsec_xs)
		ymin = 0
		ymax = pl.amax(self.imdata)
		xsec_bounds = [xmin,xmax,ymin,ymax]
				
		# Redraw cross section plot

		self.xsec_line.set_data(xsec_xs,xsec_ys)
		self.xsec_axes.axis(xsec_bounds)
		self.xsection.canvas.draw()
		
		self.DrawscatrLine(xmin=xmin)
	
	def update_result(self):
		
		# Generate coordinates at which to find image values
		n = 10000
		
		x0 = self.line.get_data()[0][0]
		x1 = self.line.get_data()[0][1]
		y0 = self.line.get_data()[1][0]
		y1 = self.line.get_data()[1][1]
		
		L = pl.sqrt((x1-x0)**2 + (y1-y0)**2)		
		
		xs,ys = GetLinePoints(n,x0,x1,y0,y1)
		
		L = L * bfp_scaling_factor
		
		res_xs = pl.linspace(-L/2.,L/2,n)
		res_ys = map_coordinates(self.unNoise(),[ys,xs], order=1)
		
		# Set new cross section plot axes limits
		xmin = -L/2. #pl.amin(xsec_xs)
		xmax = L/2. #pl.amax(xsec_xs)
		ymin = 0
		ymax = pl.amax(self.imdata)
		res_bounds = [xmin,xmax,ymin,ymax]
				
		# Redraw cross section plot

		self.result_line.set_data(res_xs,res_ys)
		self.result_axes.axis(res_bounds)
		self.result_fig.canvas.draw()
	
	### Keyboard Event Handler ###
	def keydown(self, event):
		
		if event.key == 'shift': self.movering = True
		if event.key == 'control':
			self.center = (0,0)
			self.rad1 = 0
			self.rad2 = 0
			self.DrawRing()

			
		if event.key == 'enter':
			self.pwr.append(self.getpwr())
			self.pce.append(self.getpwr()/self.beampwr[self.wli-1])
			self.update_image()
			self.update_xsec()
			self.update_result()
			
		if event.key == 'escape':
			exit()
					
	def keyup(self, event):
		if event.key == 'shift': self.movering = False
		
	def xsec_keydown(self,event):
		if event.key == 'control':
			self.scatr_y0 = self.scatr_y1 = 0
			self.DrawscatrLine()
			self.update_result()
		
	### Mouse Click Event Handler ###
	def click(self, event):
		
		#print "click: x=%f, y=%f" % (event.xdata,event.ydata)
		if not self.movering: return
		if event.inaxes!=self.imaxes: return
		
		if event.button==1:
			oldcenter = self.center
			self.center = (event.xdata,event.ydata)
		
		if event.button==2: self.rad1 = pl.sqrt((event.xdata - self.center[0])**2 + (event.ydata - self.center[1])**2)
		if event.button==3: self.rad2 = pl.sqrt((event.xdata - self.center[0])**2 + (event.ydata - self.center[1])**2)
		
		if self.radii == None:
			self.inrad = min(self.rad1,self.rad2)
			self.outrad = max(self.rad1,self.rad2)
		
		if self.outrad == self.rad1 and event.button == 2: self.DrawLine(event.xdata, event.ydata)
		elif self.outrad == self.rad2 and event.button == 3: self.DrawLine(event.xdata, event.ydata)
		elif event.button == 1:
			oldxs, oldys = self.line.get_data()
			
			delx = self.center[0] - oldcenter[0]
			newx = oldxs[0] + delx
			
			dely = self.center[1] - oldcenter[1]
			newy = oldys[0] + dely
			
			self.DrawLine(newx,newy)

		self.DrawRing()
		
		self.update_xsec()
		self.update_result()
		self.getpwr()

	def xsec_click(self, event):
		
		if event.button==1: self.scatr_y0 = event.ydata
		if event.button==3: self.scatr_y1 = event.ydata
		
		self.DrawscatrLine()
		self.update_result()
		self.getpwr()
		
	def DrawCircle(self):
		
		self.imaxes.patches = []											# Remove existing shapes from figure
		
		# Filled Region
		self.ring1 = Wedge(self.center,									# Ring center
							self.outrad,								# Outer Ring Diameter
							0,360,										# range of degrees to sweep through (leave 0-360 for full circle)
							width=0,				# Thickness of ring
							linewidth=1,
							linestyle='dotted',
							edgecolor='w',
							facecolor=None
							)

		#self.ring1.set_alpha(0.2)										# Set transparency (0-1)
		self.imaxes.add_patch(self.ring1)									# Add updated circle to figure

		self.imaxes.figure.canvas.draw()
		print 'center:', self.center
		print 'inrad:', self.inrad
		print 'outrad:', self.outrad
		print ''
		print self.getpwr()
		print ''
		
	def DrawLine(self,xdata,ydata):
		
		delx = xdata - self.center[0]
		dely = ydata - self.center[1]
		
		self.line.set_data([self.center[0]-delx,xdata],[self.center[1]-dely,ydata])
		self.imaxes.figure.canvas.draw()
		
	def DrawscatrLine(self,xmin=None):
		
		# Left line
		
		if xmin!=None:
			nx0 = xmin + (self.outrad-self.inrad) * bfp_scaling_factor
			nx1 = xmin
			ny0 = self.scatr_y0
			ny1 = self.scatr_y1
		else:
			nx0 = self.xsec_scatrlineL.get_data()[0][0]
			nx1 = self.xsec_scatrlineL.get_data()[0][1]
			ny0 = self.scatr_y0
			ny1 = self.scatr_y1
		
		self.xsec_scatrlineL.set_data([nx0,nx1],[ny0,ny1])

		# Right line
		
		nx0 *= -1
		nx1 *= -1
		
		self.xsec_scatrlineR.set_data([nx0,nx1],[ny0,ny1])

		self.xsection.canvas.draw()

	def DrawRing(self):
		self.imaxes.patches = []											# Remove existing shapes from figure
		
		# Filled Region
		self.ring1 = Wedge(self.center,									# Ring center
							self.outrad,										# Outer Ring Diameter
							0,360,										# range of degrees to sweep through (leave 0-360 for full circle)
							width=self.outrad-self.inrad,							# Thickness of ring
							linewidth=1,
							edgecolor='none',
							facecolor='g'
							)
		# Edges				
		self.ring2 = Wedge(self.center,									# Ring center
							self.outrad,										# Outer Ring Diameter
							0,360,										# range of degrees to sweep through (leave 0-360 for full circle)
							width=self.outrad-self.inrad,							# Thickness of ring
							linewidth=3,
							edgecolor='g',
							facecolor='none'
							)		
		
		self.ring1.set_alpha(0.2)										# Set transparency (0-1)
		self.imaxes.add_patch(self.ring1)									# Add updated circle to figure
		self.imaxes.add_patch(self.ring2)									

		self.imaxes.figure.canvas.draw()
		print 'center:', self.center
		print 'inrad:', self.inrad
		print 'outrad:', self.outrad
		print self.getpwr()
		print ''
	
	def unNoise(self):
		'''
		applies filters to raw image data based on ring placement, scattering subtraction, and average background noise
		located near the (0,0) point.
		'''
		imrows, imcolumns = pl.shape(self.imdata)
		x0,y0 = self.center
		a = pl.arange(imrows*imcolumns)
		imflat = self.imdata.flatten()
		ringmask = (((divmod(a,imcolumns)[1]-x0)**2 + (divmod(a,imcolumns)[0]-y0)**2) < self.outrad**2) * \
				(((divmod(a,imcolumns)[1]-x0)**2 + (divmod(a,imcolumns)[0]-y0)**2) > self.inrad**2)
		
		#radial position with 0 at inner ring boundary
		ringpos = (pl.sqrt((divmod(a,imcolumns)[1]-x0)**2 + (divmod(a,imcolumns)[0]-y0)**2) - self.inrad)
		
		#slope of scattering line in xsection plot
		slope = (self.scatr_y1 - self.scatr_y0)/((self.outrad-self.inrad))
		
		sctrmask = slope*ringpos + self.scatr_y0
		
		SPPImFlat = (imflat - sctrmask) * ringmask
		
		# remove negative values
		SPPImFlat = SPPImFlat * (SPPImFlat >= 0)
		
		SPPimage = SPPImFlat.reshape(imrows,imcolumns)

		return SPPimage
		
	def iterate(self):
		ipwr = self.getpwr()
		self.pwr.append(ipwr)
		self.pce.append(ipwr/beampwr[i])

	def getpwr(self):
		pwr = pl.sum(self.unNoise())
				
		print 'std pwr:', pwr

		return pwr
					
class GaussianBuilder:
	'''
	Was written to check Gaussian-ness of excitation beam profiles.  I don't think this class is 
	implimented in any relevant functions anymore.
	'''
	def __init__(self, line, image, xsection, gaussian):
		self.line = line
		self.image = image
		self.xsec = xsection
		self.gauss = gaussian
		self.xs = list(line.get_xdata())
		self.ys = list(line.get_ydata())
		self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
	
		# Reset axes limits for data plot
		xmin = 0
		xmax = self.image.shape[1]
		ymin = 0
		ymax = self.image.shape[0]
		dataplot_bounds = [xmin,xmax,ymin,ymax]
		self.line.axes.axis(dataplot_bounds)
	
		#self.xsection_fig = pl.figure()
		#self.xsection_axes = self.xsection_fig.add_subplot(111)
		#self.xsection, = self.xsection_axes.plot([],[])
		
		# interpolate image
		#self.image_int = interp2d(pl.arange(len(image[0,:])),pl.arange(len(image[:,0])),image,kind='cubic')
		
	### Mouse Click Event Handler ###
	def __call__(self, event):
		
		print "click: x=%f, y=%f" % (event.xdata,event.ydata)
		
		if event.inaxes!=self.line.axes: return
		
		#if len(self.xs) == 2:
		#	self.xs = []
		#	self.ys = []
			
		#self.xs.append(event.xdata)
		#self.ys.append(event.ydata)
		
		#event.button 1=left, 2=wheel, 3=right
		if event.button==1:
			self.xs[0] = event.xdata							
			self.ys[0] = event.ydata

		if event.button==3:
			self.xs[1] = event.xdata							
			self.ys[1] = event.ydata

		self.line.set_data(self.xs, self.ys)
		self.line.figure.canvas.draw()
		
		# Plot cross section of image data
		if len(self.xs) == 2:
			
			# Generate coordinates at which to find image values
			n = 1000
			
			x0 = self.xs[0]
			x1 = self.xs[1]
			y0 = self.ys[0]
			y1 = self.ys[1]
			
			L = pl.sqrt((x1-x0)**2 + (y1-y0)**2)
			print "raw length: %0.2f" % L
			print "length: %0.2e" % (L * real_scaling_factor)
			
			
			xs,ys = GetLinePoints(n,x0,x1,y0,y1)
			
			# Determine coordinate pairs to display in cross section plot
			L = L * real_scaling_factor
			
			xsec_xs = pl.linspace(0,L,n)
			xsec_ys = map_coordinates(self.image,[ys,xs])
			
			lpf_ys = SimpleLP(xsec_xs,xsec_ys,2e5,plotfft=False)
			gauss_ys,width = GaussianFit(xsec_xs,xsec_ys)
			
			print 'Beam radius (1/e^2):', 2*width
			
			# Set new cross section plot axes limits
			xmin = 0 #pl.amin(xsec_xs)
			xmax = L #pl.amax(xsec_xs)
			ymin = 0
			ymax = pl.amax(self.image)
			xsec_bounds = [xmin,xmax,ymin,ymax]
			
			# Redraw cross section plot

			self.xsec.set_data(xsec_xs,lpf_ys)
			self.gauss.set_data(xsec_xs,gauss_ys)
			self.xsec.axes.axis(xsec_bounds)
			self.xsec.figure.canvas.draw()
			


##############################################################################################################






##############################################################################################################
'''
IMAGE LOADING

This module supports loading several different file formats, corresponding to the output from
different software systems:

FITS (.fit) - SXV-H9 (Back focal plane images)
TIFF (.tif) - Winspec/Pixis
JPG			- WinTV/Sony CCD (real plane images)
'''

def LoadFITS(fname):
	hdulist = pyfits.open(fname, ignore_missing_end=True)
	image = hdulist[0].data
	
	print 'Loading -', fname
	print 'Image Dimensions:', pl.shape(image)
	print 'Maximum Counts:', pl.amax(image)
	print ''

	return image
	
def LoadJPG(fname):
	image = pltImg.imread(fname)
	image = image[:,:,0]												#flatten to one color channel
	
	return image
	
def LoadTiff(fname):
	image = Tiff.imread(fname)
	return image

def LoadPixisCSV(fname):
	image = np.loadtxt(fname, delimiter=',')
	print np.shape(image)
	exit()
	
def PlotImage(image,vmax=None,vmin=None):
	im = pl.imshow(image, cmap='hot', vmax=vmax, vmin=vmin)
	#pl.imshow(pl.log(image), cmap='hot')
	pl.colorbar()
	
	return im
	
def CleanImage(im, bkgr_rad=100):
	'''
	Removes DC background level from image.  Averages value in lower-right corner of image within user-specified radius
	and subtracts from total image matrix.
	'''
	center = (200,200)
	
	im = im.astype(float)
	
	imrows, imcolumns = pl.shape(im)
	a = pl.arange(imrows*imcolumns)
	imflat = im.flatten()
	bkgrmask = (((divmod(a,imcolumns)[1]-center[1])**2 + (divmod(a,imcolumns)[0]-center[0])**2) < bkgr_rad**2)	
	
	immasked = imflat*bkgrmask
	noise = pl.sum(immasked) / pl.sum(bkgrmask)
	print 'sum of background counts:', pl.sum(immasked)
	print 'area of quarter circle:', pl.pi*bkgr_rad**2/4.
	print 'average noise (in counts):', noise
	
	im -= noise
	
	#Bulldozer noise flattener:
	threshold = 200
	im = im * (im > threshold)
		
	return im
	
def RemoveDark(image,wl):
	'''
	NOT CURRENTLY USED
	
	Loads filename of form wl + "_dark." + filetype and subtracts from image.  Returns image data array.
	
	eg. 700.fit -> 700_dark.fit
	'''
	
	darkname = str(wl) + '_dark' + filetype
	
	if filetype == '.fit': dark = LoadFITS(darkname)
	elif filetype == '.TIF': dark = LoadTiff(darkname)
	
	image = image - dark
	
	return image

def Pixels2Power(image,wl=None,inttime=None):
	if wl == None: wl = input('Enter wavelength (nm):')
	if inttime == None: inttime = input('Enter Integration Time (ms):')
	
	# Load correction factor from calibration file
	with open(CALPATH, 'rb') as f:
		reader=csv.reader(f)
		for row in reader:
			if abs(float(row[0]) - wl) < 0.000000001: calfactor = float(row[1])
	
	# Ensure calibration factor exists		
	try:
		calfactor
	except NameError:
		print "Calibration factor does not exist for desired wavelength (%u)" % wl
		exit()
	
	image = image * calfactor / inttime									# Converts pixel data from counts into nW
	
	return image

def LoadPowerImage(fname,wl=None,inttime=None):
	
	"""
	Gets image data from a FITS file and converts pixel data into units of nW based on existing calibration data
	"""
	
	if wl == None: wl = input('Enter wavelength (nm):')
	if inttime == None: inttime = input('Enter Integration Time (ms):')

	if os.path.splitext(fname)[-1] == '.fit': image = LoadFITS(fname)
	elif os.path.splitext(fname)[-1] == '.TIF': image = LoadTiff(fname)

	#image = RemoveDark(image,wl)
	image = CleanImage(image)
	
	image = Pixels2Power(image,wl,inttime)
	
	return image, wl, inttime
	

##############################################################################################################





##############################################################################################################
'''
CALBRATION TOOLS
'''

def Calibrate(datafile):
	
	"""
	Run within a directory containing FITS files corresponding to the back focal plane image of a blank glass slide
	at different wavelengths.  The filenames should be formatted as (wavelength in nm).fit (eg. '570.fit').  If using 
	the Pixis CCD the files should be saved as TIFs.
	
	Loads a spreadsheet file containing data in 4 columns (wavelength, total beam power, ambient/dark power, integration time -- no headers).
	
	Sequentially loads FITS files and integrates total counts.   The power for a given pixel is calculated as Power = counts * cal_factor / int_time,
	and thus using this data the cal_factor for each wavelength is determined.
	"""
	
	wls = pl.array([])													# wavelength in nm
	beampwr = pl.array([])												# measured fianium beam power in nW
	darkpwr = pl.array([])												# measured ambient light power in nW
	inttime = pl.array([])												# sxv-h9 integration time in ms
	counts = pl.array([])												# total counts in a given FITS file	
		
	# Load wavelength,power,integration time data from file	
	with open(datafile,'rb') as df:
		reader=csv.reader(df)
		for row in reader:
			wls = pl.append(wls,float(row[0]))
			beampwr = pl.append(beampwr,float(row[1]))
			darkpwr = pl.append(darkpwr,float(row[2]))
			inttime = pl.append(inttime,float(row[3]))
	
	# Correct measured beam power for ambient light level
	beampwr = beampwr - darkpwr
	
	# Collect FITS files from current directory
	path = os.getcwd()
	for f in os.listdir(path):
		if os.path.splitext(f)[-1] == filetype:
			if filetype == '.TIF': im = LoadTiff(f)
			elif filetype == '.fit': im = LoadFITS(f)
			
			im = CleanImage(im)
			
			counts = pl.append(counts,(pl.sum(im)))
				
	# Compute calibration factor
	calfactor = beampwr * inttime / counts
	
	# Output calibration data as a csv
	with open('calibration.csv', 'wb') as f:
		writer = csv.writer(f)
		for i,wl in enumerate(wls):
			writer.writerow([wl,calfactor[i]])
	
	pl.plot(wls,calfactor)
	pl.show()
			
def Grating(fname, mode='pixis'):
	
	"""
	Loads FITS image and plots cross-sections of main image overlaid with markers indicating predicted diffraction grating maxima.
	To calibrate BFP distances, load an image from a known grating with the calibration multiplier set to 1. Then,
	use the cross section plot to determine the uncalibrated distance between maxima.  Enter this value
	in the top "global parameters" section.  Upon reloading the image, the grating maxima should coincide
	with the dashed lines in the cross section plot.  The height of the sinc amplitude envelope must be set manually.
	"""
	#Experimental parameters -- change for current analysis
	wl = 555/index
	spacing = 2500														# Periodicity of grating in nm
	width = 300															# pitch of grating lines in nm
	
	fname = os.getcwd() + '/' + fname
	
	if mode == 'sxv':
		image = LoadFITS(fname)
	elif mode == 'pixis':
		image = LoadTiff(fname)
		
	print 'image dimensions:', image.shape
	
	PlotImage(image)
	
	#Create line overlay
	pl.gca().hold(True)
	line, = pl.plot([0,0], [0,0])  # empty line
	
	#Create Cross Section figure
	xsection_fig = pl.figure()
	xsection_axes = xsection_fig.add_subplot(111)
	xsection, = pl.plot([],[],lw='3')
	
	# Calculate sinc() envelope effect due to grating pitch
	p = pl.linspace(-1,1,500)
	sinc = (pl.sin(pl.pi*width*p/wl) / (pl.pi*width*p/wl))**2
	sinc_coef = 2700 / (pl.sin(pl.pi*width/spacing)/(pl.pi*width/spacing))**2
	sinc = sinc*sinc_coef
	
	#Plot diffraction grating markers
	for i in range(-10,10): xsection_axes.axvline((i+1)*wl/spacing, color='k',linestyle='--',lw='1')
	xsection_axes.plot(p,sinc)
	
	#launch clickable image object
	csp = CrossSectionPlot(line,image,xsection)
	
	pl.xlim(0,len(image[0,:]))
	pl.ylim(0,len(image[:,0]))

	pl.show()
	
##############################################################################################################





##############################################################################################################
'''
MAIN ROUTINES
'''

def Xsection(fname):
	
	"""
	Loads image and plots cross-sections of main image
	"""
	
	fname = os.getcwd() + '/' + fname
	
	if os.path.splitext(fname)[-1] == '.fit':
		image = LoadPowerImage(fname)[0]
		
	elif os.path.splitext(fname)[-1] == '.jpg' or os.path.splitext(fname)[-1] == '.jpeg':
		image = LoadJPG(fname)
	
	elif os.path.splitext(fname)[-1] == '.TIF':
		image = LoadPowerImage(fname)[0]
	
	print 'image dimensions:', image.shape
	PlotImage(image)
	
	#Create line overlay
	pl.gca().hold(True)
	line, = pl.plot([0,0], [0,0])  # empty line
	
	#Create Cross Section figure
	xsection_fig = pl.figure()
	xsection_axes = xsection_fig.add_subplot(111)
	xsection, = pl.plot([],[],lw='3')
	
	#launch clickable image object
	csp = CrossSectionPlot(line,image,xsection)
	
	pl.xlim(0,len(image[0,:]))
	pl.ylim(0,len(image[:,0]))
	
	pl.show()

	
def LRM_GetPlasmons_one(fname):
	
	"""
	Used to debug SPP calculation.  Possibly out of date.  
	"""
	
	im, wl, inttime = LoadPowerImage(fname)
	
	print "--- Instructions ---"
	print """
Hold shift while clicking to select integration region.
Right Button - choose ring center;
Middle Button - choose radius 1;
Left Button - choose radius 2;

Press Ctrl to reset;
Press Enter when finished.
		"""
	
	im = im / np.amax(im)
	PlotImage(im)

	#launch clickable image object
	pl.gca().hold(True)
	
	if wl==None: cb = CircleBuilder(pl.gca(),im)
	else:
		radii=GetSPPbounds(wl, width=0)
		#cb = CircleBuilder(pl.gca(),im)
		cb = CircleBuilder(pl.gca(),im,radii=radii)
		
	pl.show()
	
	## After figure closes (user presses enter)
	#center = cb.center
	#outrad = max(cb.rad1,cb.rad2)
	#inrad = min(cb.rad1,cb.rad2)
	
	#pwr = 0
	#for row in im:
		#for p,col in enumerate(row):
			#if RingTest(row,col,center,inrad,outrad): pwr+=p
	return 0
	
	
def LRM_GetPlasmons(fname):
	
	"""
	Given a csv containing a list of wavelengths, incident beam powers, and ambient powers,
	sequentially opens corresponding FITS images within the same directory and integrates the power
	contained within a ring (the ring center is defined for the first image and held constant for
	subsequent images; the radii are calculated based on wavelength and substrate material).
	
	Data are reported in a plot and saved to a csv report within the same directory as fname (the original list).
	"""
	
	wls = pl.array([])													# wavelength in nm
	beampwr = pl.array([])												# measured fianium beam power in nW
	darkpwr = pl.array([])												# measured ambient light power in nW
	inttime = pl.array([])												# sxv-h9 integration time in ms
	pwr = pl.array([])													# integrated SPP power in nW
	pce = pl.array([])													# Relative Power Conversion Efficiency, pwr/(beampwr-darkpwr)
		
	# Load wavelength,power,integration time data from file	
	with open(fname,'rb') as df:
		reader=csv.reader(df)
		for row in reader:
			wls = pl.append(wls,float(row[0]))
			beampwr = pl.append(beampwr,float(row[1]))
			darkpwr = pl.append(darkpwr,float(row[2]))
			inttime = pl.append(inttime,float(row[3]))
	
	# Correct measured beam power for ambient light level
	beampwr = beampwr - darkpwr
	
	# Collect FITS files from current directory
	path = os.path.dirname(fname)

	ip = InteractivePlot(fname,beampwr,wls,inttime,radii=None)
	pl.show()
	
	pce = ip.pce	
	pwr = ip.pwr	
	
	# Create pce plot
	pcefig = pl.figure()
	pl.plot(wls,pce,'r')
	pl.savefig(path + 'pce_plot.png')		
	
	pcefig = pl.figure()
	pl.plot(wls,pwr,'b')
	pl.savefig(path + 'pwr_plot.png')		
	
	pcefig = pl.figure()
	pl.plot(wls,beampwr,'g')
	pl.savefig(path + 'beampwr_plot.png')		
	
	# Output data as a csv
	with open(path + 'results.csv', 'wb') as f:
		writer = csv.writer(f)
		for i,wl in enumerate(wls):
			writer.writerow([wl,pwr[i],pce[i]])
		

##############################################################################################################





##############################################################################################################
'''
SCRAP CODE

The following routines were used during development but are obsolete.
'''


'''
def test_ring_area(fname):
	
	"""
	Given a csv containing a list of wavelengths, incident beam powers, and ambient powers,
	sequentially opens corresponding FITS images within the same directory and integrates the power
	contained within a ring (the ring center is defined for the first image and held constant for
	subsequent images; the radii are calculated based on wavelength and substrate material).
	
	Data are reported in a plot and saved to a csv report within the same directory as fname (the original list).
	"""
	
	wls = pl.array([])													# wavelength in nm
	beampwr = pl.array([])												# measured fianium beam power in nW
	darkpwr = pl.array([])												# measured ambient light power in nW
	inttime = pl.array([])												# sxv-h9 integration time in ms
	widths = pl.array([1,5,10,20,30,100])
		
	# Load wavelength,power,integration time data from file	
	with open(fname,'rb') as df:
		reader=csv.reader(df)
		for row in reader:
			wls = pl.append(wls,float(row[0]))
			beampwr = pl.append(beampwr,float(row[1]))
			darkpwr = pl.append(darkpwr,float(row[2]))
			inttime = pl.append(inttime,float(row[3]))
	
	pwr = pl.zeros((len(wls),len(widths)))								# integrated SPP power in nW
	pce = pl.zeros((len(wls),len(widths)))								# Relative Power Conversion Efficiency, pwr/(beampwr-darkpwr)
	
	#pce[0,1]=1
	#pce[1,0]=2
	#print pce
	#exit()
	
	# Correct measured beam power for ambient light level
	beampwr = beampwr - darkpwr
	
	# Collect FITS files from current directory
	path = os.path.dirname(fname)
	for i,wl in enumerate(wls):		
		wl = int(wl)
		im = LoadPowerImage(path + str(wl) + '.fit',wl,inttime[i])[0]
		#im = LoadFITS(path + str(wl) + '.fit')
		
		for wi,w in enumerate(widths):
			# Calculate SPP Arc bounds
			radii=GetSPPbounds(wl,w)
					
			# Prompt user to select center point for integration
			try:
				cb
			except:
				PlotImage(im)

				# launch clickable image object
				pl.gca().hold(True)
				cb = CircleBuilder(pl.gca(),im,radii=radii)
				#cb = CircleBuilder(pl.gca(),im)
				
				pl.show()
				center = cb.center
			else:
				cb.outrad, cb.inrad = radii
				cb.image = im
				#pl.savefig(path + 'int_region_' + str(wl) + '.png')
			
			
			ipwr = cb.getpwr()
			pwr[i,wi] = ipwr
			pce[i,wi] = ipwr/beampwr[i]
			
			savepath = 'Ring Plots/wl.' + str(wl) + '_width.' + str(w) + '.png'
			
			if not os.path.exists(os.getcwd() + '/Ring Plots/'): os.makedirs(os.getcwd() + '/Ring Plots/')
			PlotRing(im,center,radii,savepath)
			
	# Create pce plot
	pcefig = pl.figure()
	for wi,w in enumerate(widths):
		pce[:,wi] = pce[:,wi]/pl.amax(pce[:,wi])
		pl.plot(wls,pce[:,wi])
	pl.legend(widths)
	pl.savefig(path + 'testring_pce_plot.png')		
		
	
	## Output data as a csv
	#with open(path + 'testring_results.csv', 'wb') as f:
		#writer = csv.writer(f)
		#for i,wl in enumerate(wls):
			#writer.writerow([wl,pwr[i],pce[i]])
		#writer.writerow(['Center coordinates=(%u,%u)' % center])
		
def PlotRing(image,center,radii,savepath):
	
	outrad,inrad = radii
	
	fig = pl.figure()
	ax = fig.add_subplot(111)
	ax.imshow(image,cmap='hot')
	pl.colorbar()
	
	# Filled Region
	ring1 = Wedge(center,									# Ring center
						outrad,										# Outer Ring Diameter
						0,360,										# range of degrees to sweep through (leave 0-360 for full circle)
						width=outrad-inrad,							# Thickness of ring
						linewidth=1,
						edgecolor='none',
						facecolor='g'
						)
	# Edges				
	ring2 = Wedge(center,									# Ring center
						outrad,										# Outer Ring Diameter
						0,360,										# range of degrees to sweep through (leave 0-360 for full circle)
						width=outrad-inrad,							# Thickness of ring
						linewidth=3,
						edgecolor='g',
						facecolor='none'
						)		
	
	ring1.set_alpha(0.2)										# Set transparency (0-1)
	ax.add_patch(ring1)									# Add updated circle to figure
	ax.add_patch(ring2)									

	pl.savefig(savepath)
	pl.close()
	return
	
def BeamWidth(fname):
	
	fname = os.getcwd() + '/' + fname
	image = LoadJPG(fname)
	print 'image dimensions:', image.shape
	
	PlotImage(image)
	
	#Create line overlay
	pl.gca().hold(True)
	line, = pl.plot([0,0], [0,0])  # empty line
	
	#Create Cross Section figure
	xsection_fig = pl.figure()
	xsection_axes = xsection_fig.add_subplot(111)
	xsection, = pl.plot([],[],lw='1',color='r')
	gaussian, = pl.plot([],[],lw='1',color='b')
	
	#launch clickable image object
	gb = GaussianBuilder(line,image,xsection,gaussian)
	
	pl.xlim(0,len(image[0,:]))
	pl.ylim(0,len(image[:,0]))
	
	pl.show()
	
def testCleanImage():
	
	im = pl.ones((1e4,1e4),dtype=float)
	
	CleanImage(im,1e3)
	
def PowerLinearity():
	
	power = []
	counts = []
	
	for f in os.listdir(os.getcwd()):
		if os.path.splitext(f)[-1] == '.fit':
			p = os.path.basename(f).split('.fit')[0]
			power.append(float(p))
			
			im = LoadFITS(f)
			counts.append(pl.amax(im))
	
	counts = pl.array(counts,dtype=float)
	power = pl.array(power,dtype=float)
	
	m,b = putil.LinFit(power,counts,b=0)
	fit_counts = m*power + b
	pl.plot(power,counts,'ro')
	pl.plot(power,fit_counts,'b-')
	
	print 'Slope:',m
	
	pl.xlabel('Incident Beam Power (nW)')
	pl.ylabel('Maximum Image Counts')
	pl.show()

def TimeLinearity():
	
	time = []
	counts = []
	
	for f in os.listdir(os.getcwd()):
		if os.path.splitext(f)[-1] == '.fit':
			t = os.path.basename(f).split('.fit')[0]
			time.append(float(t))
			
			im = LoadFITS(f)
			counts.append(pl.amax(im))
	
	counts = pl.array(counts,dtype=float)
	time = pl.array(time,dtype=float)
	
	m,b = putil.LinFit(time,counts,b=0)
	fit_counts = m*time + b
	pl.plot(time,counts,'ro')
	pl.plot(time,fit_counts,'b-')
	
	print 'Slope:',m
	
	pl.xlabel('Integration Time (ms)')
	pl.ylabel('Maximum Image Counts')
	pl.show()
	
'''

##############################################################################################################





##############################################################################################################
'''
IMAGE PLOTTERS

These functions are handy since Plasmonaut does not have instances of Winspec or the SXV software.
'''
def viewer(fname):
	'''
	Plots fname and nothing else.
	'''
	
	fname = os.getcwd() + '/' + fname
	image = LoadFITS(fname)
	
	PlotImage(image)

	pl.show()

def Powerviewer(fname):
	'''
	Plots fname scaled to power.
	'''
	
	fname = os.getcwd() + '/' + fname
	image = LoadPowerImage(fname)[0]
	
	PlotImage(image)

	pl.show()
	
##############################################################################################################




##############################################################################################################
'''
COMMAND LINE INTERFACE

This is the main function run by the module.  Command line arguments are 
interpreted here.
'''

def ImageProcessing():
	
	if len(sys.argv) > 1: fname = sys.argv[1]
	else: print 'LRM Image Analysis: arg 1 must be an image filename.'; sys.exit()
	
	if fname == 'calibrate':
		if len(sys.argv) > 2: datafile = sys.argv[2]
		else: print 'LRM Image Analysis - Calibration: arg 2 must be a csv filename.'; sys.exit()
		
		Calibrate(datafile)
		
		sys.exit()

	print fname	
	
	'''
	Change uncommented function to switch modes of operation.  LRM_GetPlasmons is the 
	function for analyzing SPP data.
	'''
	#Xsection(fname)
	LRM_GetPlasmons(fname)
	#test_ring_area(fname)
	#LRM_GetPlasmons_one(fname)
	#Grating(fname)
	#BeamWidth(fname)
	#LoadPixisCSV(fname)
	#Powerviewer(fname)


if __name__ == '__main__':
	
	ImageProcessing()
	
	# Obsolete
	
	#PowerLinearity()
	#TimeLinearity()
