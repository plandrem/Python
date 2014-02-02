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

def dipoleRadiation(wl,amp,xo,yo,x,y):
  '''
  calculates Fz at (x,y) from dipole of amplitude amp centered at (xo,yo) 
  '''

  r = sqrt((x-xo)**2 + (y-yo)**2)

  k = 2*pi/wl

  f = amp * exp(1j*k*r) / r

  return f

def dipoleRadiation3D(wl,P,r):
  '''
  Computes the radiated electric field, E, at a position r from the dipole coordinates.  P,r, and E are all 3-vectors
  '''

  if len(np.shape(r)) == 3: #2D array of vectors
    dim = 2
    axis = 1
    shape = r.shape

    # flatten r
    n = r.shape[0]*r.shape[1]
    r = r.reshape(n,3)

    # create copies of P for each vector in r
    P = np.tile(P,(n,1))

  else:
    dim =1
    axis=0

  
  k = 2*pi/wl
  if dim==2: 
    d = np.array([np.linalg.norm(i) for i in r])
    d = np.vstack([d,d,d]).T
  else:
    d = np.linalg.norm(r)

  rxP = np.cross(r,P)
  rxrxP = np.cross(r,rxP)

  rdotP = np.sum(r*P,axis=axis)
  if dim == 2: rdotP = rdotP.reshape(len(rdotP),1)

  if d == 0:
    E = np.nan
  else:
    E = exp(1j*k*d)/d**3 * (k**2*rxrxP + (1-1j*k*d)/d**2 * (d**2*P - 3*r*rdotP))

  if dim == 2:
    E = E.reshape(shape)

  return E


def test_dipoleRadiation3D():
  '''
  Plots the Electric field in the x,y plane from both a z and y oriented dipole.
  Right now, the y-dipole plots |E| -- ultimately this should be a vector plot to highlight
  the loop pattern.
  '''

  wl = 2

  #z-oriented dipole
  P = np.array([0,0,1])

  res = 100

  x = np.linspace(-5,5,res)
  y = np.linspace(-5,5,res)
  z = 0

  X,Y = np.meshgrid(x,y)
  Z = X*0

  r = np.array([X,Y,Z])
  r = np.swapaxes(r,0,2)

  E = dipoleRadiation3D(wl,P,r)

  Ez = E[:,:,2]

  vmax = 10 #np.amax(abs(Ez))
  vmin = -vmax
  plt.imshow(Ez.real,extent=putil.getExtent(x,y),cmap='RdBu',vmax=vmax,vmin=vmin)
  plt.colorbar()


  # y-oriented dipole
  P = np.array([0,1,0])
  E = dipoleRadiation3D(wl,P,r)


  Etot = np.array([np.linalg.norm(i) for i in E.reshape(res**2,3)]).reshape(res,res)

  plt.figure()
  plt.imshow(Etot.T,extent=putil.getExtent(x,y),cmap='hot',vmax=vmax,vmin=0)
  # plt.imshow(E[:,:,1].real.T,extent=putil.getExtent(x,y),cmap='RdBu',vmax=vmax,vmin=vmin)
  plt.colorbar()

  plt.show()

def addDipoles3D(wl,dipoles,Ps,target):
  '''
  given a list of dipole coordinates and polarizations (Ps), calculate the vector E field at target
  coordinates.
  '''

  E = np.zeros(3,dtype=complex)

  for i,d in enumerate(dipoles):
    r = target - d

    E += dipoleRadiation3D(wl,Ps[i],r)

  return E

def test_addDipoles3D():

  wl = 1

  d1 = [-wl/4.,0,0]
  d2 = [ wl/4.,0,0]

  ds = np.array([d1,d2])

  p1 = [0,0,1]
  p2 = [0,0,1]

  ps = np.array([p1,p2])

  res = 100
  xs = np.linspace(-5,5,res)
  ys = np.linspace(-5,5,res)

  targets = np.array([[i,j,0] for i in xs for j in ys])

  E = [addDipoles3D(wl,ds,ps,t) for t in targets]
  E = np.array(E)
  Ez = E[:,2].reshape(res,res)

  vmax = 50
  vmin = -vmax
  plt.imshow(Ez.real.T,extent=putil.getExtent(xs,ys),cmap='RdBu',vmax=vmax,vmin=vmin)
  plt.show()



  
def addDipoles(wl, dip_x, dip_y, amps, x, y):
  '''
  given a list of dipole coordinates and amplitudes, find the total field at a given point, (x,y)
  '''

  field = lambda xo,yo,amp: dipoleRadiation(wl,amp,xo,yo,x,y)

  Fzs = map(field,dip_x,dip_y,amps)

  Fz = sum(Fzs)

  return Fz

def test_addDipoles():
  
  wl = 400

  dip_x = np.array([0,0])
  dip_y = np.array([-wl/4.,wl/4.])

  amps = np.array([-1,1])

  x = np.linspace(-4*wl,4*wl,100)
  y = np.linspace(-4*wl,4*wl,100)

  X,Y = np.meshgrid(x,y)

  Fz = addDipoles(wl,dip_x,dip_y,amps,X,Y)

  print Fz.shape

  plt.imshow(Fz.real,extent=putil.getExtent(x,y))

  plt.show()

def FF3D(wl, dipoles, Ps, R=1e9,res=10):
  '''
  gets sum of dipole fields on a circle of radius R (in nm)
  '''

  theta = np.linspace(0,2*pi,res)

  x = R*cos(theta)
  y = R*sin(theta)
  z = x*0

  target = np.vstack((x,y,z)).T

  E = [addDipoles3D(wl,dipoles,Ps,t) for t in target]
  E = np.array(E)

  Ez = E[:,2]

  pwr = abs(Ez)**2

  return theta, pwr

def FF(wl, dip_x, dip_y, amps, R=1e9,res=500):
  '''
  gets sum of dipole fields on a circle of radius R (in nm)
  '''

  theta = np.linspace(0,2*pi,res)

  x = R*cos(theta)
  y = R*sin(theta)

  # X,Y = np.meshgrid(x,y)

  Fz = addDipoles(wl,dip_x,dip_y,amps,x,y)

  P = abs(Fz)**2

  return theta, P

def test_FF():

  wl = 400

  dip_x = np.array([0,0])
  dip_y = np.array([-wl/2.,wl/2.])

  amps = np.array([-1,1])

  theta, P = FF(wl,dip_x,dip_y,amps)

  plt.figure()
  plt.subplot(111,polar=True)
  plt.gca().grid(True)
  plt.plot(theta,P,color='r',lw=2)

  plt.show()
  
  
    
def main():

  wl = 400
  
  AR = 1  # height / width
  w = 208
  h = w * AR
  l = 200*wl

  m = 3
  n = 3

  res = 15
  fieldRes = 30
  zres = 20

  x = np.linspace(-w/2.,w/2.,res)
  y = np.linspace(-w/2.,w/2.,res)
  z = np.linspace(-l/2.,l/2.,zres)
  # z = [0]

  # X,Y = np.meshgrid(x,y)
  dipoles = [[i,j,k] for i in x for j in y for k in z]
  dipoles = np.array(dipoles)

  amplitude = sin(m*pi/w * (dipoles[:,0]+w/2.)) * sin(n*pi/h * (dipoles[:,1]+h/2.))
  Ps = [np.array([0,0,a]) for a in amplitude]

  # dip_x = X.flatten()
  # dip_y = Y.flatten()
  # amps  = amplitude.flatten()

  '''
  # field plot
  fieldx = np.linspace(0,4*wl+w/2.,fieldRes)
  fieldy = np.linspace(0,4*wl+h/2.,fieldRes)

  E = [addDipoles3D(wl,dipoles,Ps,[i,j,0]) for i in fieldx for j in fieldy]
  E = np.array(E)

  Ez = E[:,2].reshape(fieldRes,fieldRes)

  fieldX, fieldY = np.meshgrid(fieldx,fieldy)
  mask = (abs(fieldX) > w/2.) * (abs(fieldY) > h/2.)

  # Fz = addDipoles(wl, dip_x, dip_y, amps, fieldX, fieldY)
  
  # vmax = None
  # vmin = None

  vmax = 0.1 * np.amax(abs(np.nan_to_num(Ez.real.T)*mask))
  vmin = -vmax

  plt.figure()
  plt.imshow(Ez.real.T,extent=putil.getExtent(fieldx,fieldy),vmax=vmax,vmin=vmin,cmap='RdBu')
  plt.colorbar()
  '''

  # FF plot
  # theta, P = FF(wl,dip_x,dip_y,amps)
  theta, pwr = FF3D(wl,dipoles,Ps,R=10*wl,res=100)

  plt.figure()
  plt.subplot(111,polar=True)
  plt.gca().grid(True)
  plt.plot(theta,pwr,color='r',lw=2)

  # Dipole Arrangement
  # plt.figure()
  # plt.imshow(amplitude.reshape(res,res), extent = putil.getExtent(x,y))
  

  plt.show() 


  return 0

if __name__ == '__main__':
  # test_FF()
  main()
  # test_addDipoles3D()
