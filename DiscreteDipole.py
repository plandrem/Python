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


def dipoleRadiation(wl,P,r):
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

  elif len(np.shape(r)) == 2: # list of vectors
    dim = 1
    axis = 1
    shape = r.shape

  else:
    dim =1
    axis=0

  k = 2*pi/wl

  d = np.linalg.norm(r,axis=axis)
  d = np.tile(d,(3,1)).T

  # if axis==1: 
  #   # d = np.array([np.linalg.norm(i) for i in r])
  #   # d = np.vstack([d,d,d]).T
  #   # d = np.
  # else:
  #   d = np.linalg.norm(r)

  # compute vector products
  rxP = np.cross(r,P)
  rxrxP = np.cross(r,rxP)

  rdotP = np.sum(r*P,axis=axis)
  if axis==1: rdotP = rdotP.reshape(len(rdotP),1)

  # compute electric field
  # if d == 0:
  #   E = np.nan
  # else:
  #   E = exp(1j*k*d)/d**3 * (k**2*rxrxP + (1-1j*k*d)/d**2 * (d**2*P - 3*r*rdotP))
  E = exp(1j*k*d)/d**3 * (k**2*rxrxP + (1-1j*k*d)/d**2 * (d**2*P - 3*r*rdotP))

  # match input data arrangement
  if dim == 2:
    E = E.reshape(shape)

  return E


def test_dipoleRadiation():
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

  E = dipoleRadiation(wl,P,r)

  Ez = E[:,:,2]

  vmax = 10 #np.amax(abs(Ez))
  vmin = -vmax
  plt.imshow(Ez.real,extent=putil.getExtent(x,y),cmap='RdBu',vmax=vmax,vmin=vmin)
  plt.colorbar()


  # y-oriented dipole
  P = np.array([0,1,0])
  E = dipoleRadiation(wl,P,r)


  Etot = np.array([np.linalg.norm(i) for i in E.reshape(res**2,3)]).reshape(res,res)

  plt.figure()
  plt.imshow(Etot.T,extent=putil.getExtent(x,y),cmap='hot',vmax=vmax,vmin=0)
  # plt.imshow(E[:,:,1].real.T,extent=putil.getExtent(x,y),cmap='RdBu',vmax=vmax,vmin=vmin)
  plt.colorbar()

  plt.show()

def addDipoles(wl,dipoles,Ps,target):
  '''
  given a list of dipole coordinates and polarizations (Ps), calculate the vector E field at target
  coordinates.

  dipoles and Ps are Nx3 lists of cartesian coordinates
  target is a 3-vector
  '''

  E = np.zeros((3),dtype=complex)

  ts = np.tile(target,(dipoles.shape[0],1))
  rs = ts-dipoles

  E = np.sum(dipoleRadiation(wl,Ps,rs),axis=0)

  return E

def test_addDipoles():

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

  E = [addDipoles(wl,ds,ps,t) for t in targets]
  E = np.array(E)
  Ez = E[:,2].reshape(res,res)

  vmax = 50
  vmin = -vmax
  plt.imshow(Ez.real.T,extent=putil.getExtent(xs,ys),cmap='RdBu',vmax=vmax,vmin=vmin)
  plt.show()


def FF(wl, dipoles, Ps, R=1e9,res=10):
  '''
  gets sum of dipole fields on a circle of radius R (in nm)
  '''

  theta = np.linspace(0,2*pi,res)

  x = R*cos(theta)
  y = R*sin(theta)
  z = x*0

  target = np.vstack((x,y,z)).T

  E = [addDipoles(wl,dipoles,Ps,t) for t in target]
  E = np.array(E)

  Ez = E[:,2]

  pwr = abs(Ez)**2

  return theta, pwr
      
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
  zres = 30

  x = np.linspace(-w/2.,w/2.,res)
  y = np.linspace(-w/2.,w/2.,res)
  z = np.linspace(-l/2.,l/2.,zres)
  # z = [0]

  # X,Y = np.meshgrid(x,y)
  dipoles = [[i,j,k] for i in x for j in y for k in z]
  dipoles = np.array(dipoles)

  amplitude = sin(m*pi/w * (dipoles[:,0]+w/2.)) * sin(n*pi/h * (dipoles[:,1]+h/2.))
  Ps = [np.array([0,0,a]) for a in amplitude]


  # FF plot
  # theta, P = FF(wl,dip_x,dip_y,amps)
  theta, pwr = FF(wl,dipoles,Ps,R=10*wl,res=100)

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
  # test_addDipoles()
  # test_dipoleRadiation()
