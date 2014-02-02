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


def transcendental(n,L,h,k):
  '''
  eigenvalue finder for slab cooled by convection on both sides
  
  Inputs:
  n = number of terms to return
  L = half width of slab
  h = convection heat transfer coefficient
  k = slab conductivity
  
  Returns:
  ls = list of eigenvalues, lambda
  '''
  
  res = 1e5

  lam = np.linspace(0,(n-0.5)*pi,res)
  
  RHS = k/h*lam
  LHS = 1/tan(lam*L)
  
  xing = np.diff(np.sign(RHS-LHS))
  ls = lam[np.where(xing==2)]                     # eigenvalue solutions

  
  # plt.plot(lam[:-1]/pi,xing)
  # plt.plot(lam/pi,RHS)
  # plt.plot(lam/pi,LHS)
  # plt.ylim(0,k/h*np.amax(lam))
  
  # plt.show()

  if len(ls) != n:
    print "insufficient resolution for eigenvalues"
    print ls
    exit()

  return ls

def getCoef(To,L,lam):
  '''
  calculates coefficients associated with eigenvalues
  '''

  return 2*To/lam * sin(lam*L) / (L + 1/lam * sin(lam*L)*cos(lam*L))

def convectionSlab(L,k,h,p,c,To,n):
  '''
  calculates 1D temperature in solid slab cooled by convection.  Initial temperature profile is 
  a uniform rect function.

  Inputs:
  L - half width of slab
  k - thermal conductivity
  h - convection heat transfer coefficient
  p - density
  c - specific heat
  To - initial temperature
  n - number of terms in approximation

  Returns:
  temp(x,t)

  '''

  # get eigenvalues
  ls = transcendental(n,L,h,k)

  A = [getCoef(To,L,l) for l in ls]

  ls = np.array(ls)
  A  = np.array(A)

  a = k/(p*c)

  env = lambda t: sp.exp(-ls**2*a*t)

  temp = lambda x,t: sum(A*env(t)*cos(ls*x)) * (abs(x) <= L)

  return temp

def simpleTemp():
  '''
  plot with for loops
  '''

  L = 1
  h = 1
  k = 1
  p = 1
  c = 1
  To = 1
  n = 5

  ls = transcendental(n,L,h,k)
  A = [getCoef(To,L,l) for l in ls]

  ls = np.array(ls)
  A  = np.array(A)

  a = k/(p*c)

  env = lambda t: sp.exp(ls**2*a*t)

  x = np.linspace(-2*L,2*L,100)

  temp = np.zeros((len(ls),len(x)))

  for i in range(len(ls)):
    temp[i,:] = A[i]*cos(ls[i]*x)

  t = np.sum(temp,0)

  plt.plot(x,t)
  plt.show()


def tempPlot():
    '''
    plot temperature in slab at time t
    '''

    L = 1
    h = 1
    k = 1
    p = 1
    c = 1
    To = 1
    n = 50

    a = k/(p*c)

    temp = convectionSlab(L,k,h,p,c,To,n)

    xs = np.linspace(-2*L,2*L,1000)
    ts = np.linspace(0,8,4)

    for t in ts:
      T  = map(temp,xs,np.ones(len(xs)) * t)
      plt.plot(xs,T)

    '''
    core temperature time dependence
    '''

    ts = np.linspace(0,10,100)

    T = map(temp,np.zeros(len(ts)),ts)

    plt.figure()
    plt.plot(ts,T,lw=2)


    plt.show()
      

def main():
  # simpleTemp()
  tempPlot()
  return 0

if __name__ == '__main__':
  main()

