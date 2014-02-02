#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle,Circle

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

def cot(x):
  return 1/tan(x)

colors = ['r','b','g','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

def getR(x0,y0,x1,y1):
  '''
  Compute Euclidean distance between p0=(x0,y0) and p1=(x1,y1)
  '''
  return sqrt((x0-x1)**2 + (y0-y1)**2)
  
def getDipoles(m,n,w,h):
  '''
  Places an m x n grid of dipoles in a rectangular area of width w, height h.
  Dipoles are placed in the center of the grid tiles.
  '''
  xmax = w/2.
  ymax = h/2.
  
  a = w/(2*m)
  b = h/(2*n)
  
  d = [(i*w/m+a-xmax,j*h/n+b-ymax) for i in range(m) for j in range(n)]
  
  return d
  
def drawDipoleBox(m,n,w,h,ax):
  
  xint = w/m
  yint = h/n
  
  xmax = w/2.
  ymax = h/2.
  
  corner = (-xmax,-ymax)
  
  boundary = Rectangle(corner,w,h,facecolor='none',lw=2)
  ax.add_patch(boundary)
  
  for i in range(m):
    for j in range(n):
      
      c = (-xmax+i*xint,-ymax+j*yint)
      ax.add_patch(Rectangle(c,w/m,h/n,facecolor='none',ls='dotted'))
      
def getPhase(m,n,di):
  '''
  convert interation index into a binary toggle to correctly alternate dipole phases
  '''
  phase = (di + (n%2==0)*np.floor(di/n)) % 2
  phase = int(phase)
  
  return phase
  
def getFF(wl,w,h,m,n,theta,Ro=1):

  k = 2*pi/wl
  
  Ro *= 1e9 #convert to nm
  
  # convert to xs, ys
  x = Ro*cos(theta)
  y = Ro*sin(theta)
  
  dipoles = getDipoles(m,n,w,h)
  
  for di,d in enumerate(dipoles):
    R = getR(d[0],d[1],x,y)
    
    phi = pi*(getPhase(m,n,di))
    
    if di==0: Fz  = 1/R*exp(1j*(k*R + phi))
    else:     Fz += 1/R*exp(1j*(k*R + phi))
  
  P = abs(Fz)**2
  return P
  
def getFF_w_Snell(wl,n,w,h,p,q,theta,Ro=1,debug=False):
  
  '''
  includes phase pickup from propagation in high-index region and bend at interface
  
  NOTE - to avoid confusion with refractive index, the mode order has been relabeled with p,q in place of m,n
  '''

  k = 2*pi/wl
  
  Ro *= 1e9 #convert to nm
  
  tc = arcsin(1/n)

  # convert to xs, ys
  x = Ro*cos(theta)
  y = Ro*sin(theta)
  
  dipoles = getDipoles(p,q,w,h)
  
  Fz = np.zeros(len(theta),dtype=complex)

  for di,d in enumerate(dipoles):   # iterate over all dipole sources
    
    phi_o = pi*(getPhase(p,q,di))
    
    prop = lambda t: propagateRay(d,t,w,h)
    L_side = map(prop,theta)
    L_side = np.array(L_side,dtype = [ ('length','f4'),('coords','f4',(2)),('side','u1')])

    L        = L_side['length']                     # length propagated
    side     = L_side['side']                       # side intersected
    coords   = L_side['coords']                     # point (x,y) at which ray intersects boundary

    trans = ['transmitted' for t in theta]
    inc   = ['incident'    for t in theta]
    extTheta = map(getAngles,np.ones(len(theta))*n,theta,side,trans)  # propagation angle exterior to resonator
    incTheta = map(getAngles,np.ones(len(theta))*n,theta,side,inc  )  # propagation angle exterior to resonator

    phi_internal = k*n*L
    
    # compute x,y coordinates of point on "far field screen" reached by rays emerging from resonator
    x_ff = (coords[:,1] + Ro*(cos(extTheta)*cot(extTheta) - sin(extTheta)) - coords[:,0]*tan(extTheta)) / (cot(extTheta) - tan(extTheta))
    y_ff = tan(extTheta) * (x_ff - coords[:,0]) + coords[:,1]

    # get distance to far field point
    d_ff = sqrt((coords[:,0]-x_ff)**2 + (coords[:,1]-y_ff)**2)

    # get phase in propagation to far field
    phi_ff = k*d_ff

    # combine rays
    for ti,t in enumerate(extTheta):

      # eliminate rays propagating beyond the critical angle

      if (t.imag == 0):
        idx = putil.getClosestIndex(theta,t)
        T = putil.Transmission(incTheta[idx],n**2,1,'TE')
        Fz[idx] += T * exp(1j*(phi_ff[ti] + phi_internal[ti] + phi_o))

  
  P = abs(Fz)**2

  if debug:

    idxs = [putil.getClosestIndex(theta,t) for t in extTheta]
    ff_angles = [theta[i] for i in idxs]

    extTheta = np.array(extTheta)

    print 'initial angle', theta/pi
    print 'critical angle', tc/pi
    print 'emitted ray?', (extTheta.imag == 0)
    print
    print 'propagation length in resonator', L
    print 'intersection at res boundary', coords
    print 'external propagation angle', np.array(extTheta)/pi
    print 'far field angle bin', np.array(ff_angles)/pi
    print 'far field measurement point', (x_ff,y_ff)
    print 'far field propagation distance', d_ff
    print 
    print 'accumulated phase',(phi_ff + phi_internal + phi_o)

    plt.figure()
    plt.plot(theta/pi,np.array(extTheta)/pi  * ((extTheta.imag == 0)),'ro')
    plt.plot(theta/pi,np.array(ff_angles)/pi * ((extTheta.imag == 0)),'bo')

    # plt.figure()
    # plt.plot(theta/pi, abs(exp(1j*(phi_ff + phi_internal + phi_o))) * ((extTheta.imag == 0)), 'go')

    plt.figure()
    plt.plot(theta/pi, np.angle(exp(1j*(phi_ff + phi_internal + phi_o))) * ((extTheta.imag == 0)) / pi, 'mo')

  return P

def test_ff_w_snell():
  '''
  text
  '''

  wl = 100
  n = 2
  w = 100
  h = 100
  p = 1
  q = 1
  theta = np.array([0.58*pi])

  getFF_w_Snell(wl,n,w,h,p,q,theta,debug=True)
  
  
def getCornerAngles(dipole,w,h):
  '''
  returns theta[0-3] corresponding to the four corners of the rectangular resonator (0 = upper right, CCW)
  
  dipole - (x,y) coordinates of current dipole
  w,h - width and height of box
  '''
  a = np.array([1,-1,-1,1]) * w/2.
  b = np.array([1,1,-1,-1]) * h/2.
  
  theta = arctan((b-dipole[1])/(a-dipole[0]))
  
  # arctan returns only values from -90 to 90 degrees - convert to 0-360
  theta[1] = pi   + theta[1]
  theta[2] = pi   + theta[2]
  theta[3] = 2*pi + theta[3]
  
  
  return theta
  
def propagateRay(initCoords,theta,w,h):
  '''
  From a starting point with a given direction, determine how far a ray travels before encountering a side wall of the resonator.
  Returns the distance traveled, the coordinates of the intersection point at the boundary, and a number 0-3 indicating which wall
  was struck [right,top,left,bottom]
  '''
  
  cornerTheta = getCornerAngles(initCoords,w,h)
  
  # slope of line through dipole point
  m = tan(theta)
  
  xo = initCoords[0]
  yo = initCoords[1]
  
  horizIntersect = lambda y: 1/m*(y-yo)+xo  #returns x
  vertIntersect  = lambda x:   m*(x-xo)+yo  #returns y
  
  side = -1
  
  # Top
  if   theta >= cornerTheta[0] and theta < cornerTheta[1]:
    y = h/2.
    x = horizIntersect(y)
    side = 1

  # Left
  elif theta >= cornerTheta[1] and theta < cornerTheta[2]:
    x = -w/2.
    y = vertIntersect(x)
    side = 2
    
  # Bottom
  elif theta >= cornerTheta[2] and theta < cornerTheta[3]:
    y = -h/2.
    x = horizIntersect(y)
    side = 3

  # Right
  elif (theta >= cornerTheta[3] and theta <= 2*pi) or (theta < cornerTheta[0] and theta >= 0):
    x = w/2.
    y = vertIntersect(x)
    side = 0
    
  else:
    print 'Invalid Theta.'
    print theta*180/pi
    print cornerTheta*180/pi
    exit()
    
  d = sqrt((x-xo)**2 + (y-yo)**2)
  finalCoords = newCoords((xo,yo),d,theta)
  
  return d,finalCoords,side
  
def newCoords(coords,d,theta):
  '''
  determines new cartesian coordinates after moving a distance d at angle theta (0 points along x axis)
  '''
  
  dx = d*cos(theta)
  dy = d*sin(theta)
  
  return (coords[0] + dx, coords[1] + dy)
  
def getAngles(n,theta,side,mode='transmitted'):
  '''
  returns the new angle of propagation of a ray after leaving the resonator
  
  theta is angle from x-axis (0-360)
  b is angle of incidence
  a is angle of exit relative to x-axis
  mode = ['transmitted','incident'] returns either the transmitted angle from 0-2*pi or the incident angle from -pi/2 to pi/2 (for ref. coefs)
  '''
  
  snell = lambda t: arcsin(n*sin(t))
  
  if side==0: #right
    if theta < pi/2.:
      b = theta
      a = snell(b)
      
    if theta > pi/2.:
      b = 2*pi - theta
      a = 2*pi - snell(b)
    
  if side==1: #top
    if theta <= pi/2.:
      b = pi/2. - theta
      a = pi/2. - snell(b)
      
    if theta >  pi/2.:
      b = theta - pi/2.
      a = pi/2. + snell(b)
    
  if side==2: #left
    if theta <= pi:
      b = pi - theta
      a = pi - snell(b)
      
    if theta >  pi:
      b = theta - pi
      a = pi + snell(b)
    
  if side==3: #bottom
    if theta <= 3*pi/2.:
      b = 3*pi/2. - theta
      a = 3*pi/2. - snell(b)
      
    if theta >  3*pi/2.:
      b = theta - 3*pi/2
      a = 3*pi/2. + snell(b)
  
  if mode == 'transmitted': return a
  if mode == 'incident'   : return b

def cleanData(thetas, data):
  '''
  remove zero entries from Fz to allow plotting by a smooth line
  '''

  idx = np.nonzero(data)
  data_clean = data[idx]
  thetas_clean = thetas[idx]

  return thetas_clean, data_clean
  

''' 
def ff_plot(ax,wl,w,h,m,n,Ro=1,angle_res=100,color='r'):
  
  k = 2*pi/wl
  
  Ro *= 1e9 #convert to nm
  T  = np.linspace(0,2*pi,angle_res)
  
  # convert to xs, ys
  x = Ro*cos(T)
  y = Ro*sin(T)
  
  dipoles = getDipoles(m,n,w,h)
  
  for di,d in enumerate(dipoles):
    R = getR(d[0],d[1],x,y)
    
    phi = pi*(getPhase(m,n,di))
    
    if di==0: Fz_ff  = cos(k*R + phi)
    else:     Fz_ff += cos(k*R + phi)
      
  ax.plot(T,abs(Fz_ff)**2, color=color, lw=3)
'''
def FFplot(ax,wl,w,h,m,n,index=4,Ro=1,angle_res=100,color='r'):
  
  Ro *= 1e9 #convert to nm
  T  = np.linspace(0,2*pi,angle_res)
  
  P = getFF(wl,w,h,m,n,T,Ro=1)
  # P = getFF_w_Snell(wl,index,w,h,m,n,T,Ro=1)

  T,P = cleanData(T,P)
  
  ax.plot(T,P, color=color)
  # ax.plot(T,P, color=color, marker='o', linestyle='none')
  



def example():
  
  
  #--------------------------------#
  # Parameters
  #--------------------------------#
  
  # Simulation
  wl = 400.
  k = 2*pi/wl
  
  w = 200
  h = 800
  
  m = 1
  n = 2
  
  # Plotting
  res = 500
  xmax = 800
  
  Ro = 1. #distance to measure field (m)
  angle_res = 100

  
  #--------------------------------#
  
  
  
  # Algorithm
  
  dipoles = getDipoles(m,n,w,h)
  
  xs = ys = np.linspace(-xmax,xmax,res)
  
  X,Y = np.meshgrid(xs,ys)
  
  for di,d in enumerate(dipoles):
        
    R = getR(d[0],d[1],X,Y)
    
    phi = pi*(getPhase(m,n,di))
    
    if di==0: Fz  = 1/R*exp(1j*(k*R + phi))
    else:     Fz += 1/R*exp(1j*(k*R + phi))
  
  
  # Plot Real Field
  fig = plt.figure()
  ax  = fig.add_subplot(111)

  Fz *= -1  #flip values so plot color matches dipole color
  Fz /= np.amax(abs(Fz))

  scale = 0.01
  
  im = ax.imshow(Fz.real,extent=putil.getExtent(xs,ys),vmax=scale,vmin=-1*scale,cmap='RdBu')
  fig.colorbar(im)
  
  drawDipoleBox(m,n,w,h,ax)
  
  for di,d in enumerate(dipoles):
    
    p = getPhase(m,n,di)
    ax.add_patch(Circle(d,15,facecolor=colors[p]))

  # Power plot
  fig = plt.figure()
  ax  = fig.add_subplot(111)

  P = abs(Fz)**2
  P /= np.amax(P)
  
  im = ax.imshow(P,extent=putil.getExtent(xs,ys),vmax=sqrt(scale),cmap='hot')
  fig.colorbar(im)
  
    
    
  # Plot Farfield
  ff_fig = plt.figure()
  ff_ax  = ff_fig.add_subplot(111,polar=True)
  ff_ax.grid(True)

  FFplot(ff_ax,wl,w,h,m,n,Ro=Ro,angle_res=angle_res)
  
  print getFF(wl,w,h,m,n,0)
    
  plt.show()
  
  return 0
  
def FFsweep():
  '''
  plots the FF scattering distribution for several resonances on one polar plot
  '''
  
  # Simulation (units in nm)
  wl = 400.
  
  w = 208
  h = 208
  
  m = 1
  ns = [1]  #add values to array to plot multiple resonances
  
  # Plotting
  Ro = 1. #distance to measure field (m)
  angle_res = 10000
  
  #------------------------------------------#
  ax = plt.subplot(111)
  # ax = plt.subplot(111,polar=True)
  ax.grid(True)
  
  for i,n in enumerate(ns):
    FFplot(ax,wl,w,h,m,n,Ro=Ro,angle_res=angle_res,color=colors[i])
    
  
  plt.show()
  

if __name__ == '__main__':
  # test_ff_w_snell()
  # FFsweep()
  example()
  
  

