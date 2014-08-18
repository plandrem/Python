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
ln = sp.log

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

def SimpleResistor():
  '''
  Steady state model of the following stack:

  GST
  Al2O3
  Al
  Quartz

  2D spreading is assumed in the Al and Quartz layers. Shape factor for a NW is taken as S = 4L / ln( 8d / pi*w )
  (from Goodson, K. E., Flik, M.I., Su, L. T., and Antoniadis, D. A., 1993, "Annealing-Temperature Dependence of the 
    Thermal Conductivity of LPCVD Silicon-Dioxide Layers," IEEE Electron Device Lett., Vol. 14, pp. 490-492.)

  '''

  ln = sp.log
  
  # Layer depths (m)    
  d_gst    = 200 * 1e-9
  d_al     = 300 * 1e-9
  d_al2o3  = 20  * 1e-9
  d_quartz = 500 * 1e-9

  # Layer conductivites [W/mK]
  k_gst    = 0.4
  k_al     = 300
  k_al2o3  = 30
  k_quartz = 1

  w  = 500 * 1e-9 # width of nw
  w2 = 10000 * 1e-9 # effective width after spreading in Al layer

  L = 100e-6 # Length of device -- should get normalized out somewhere

  To = 1400

  r_gst   = d_gst  /(k_gst   * w * L)
  r_al2o3 = d_al2o3/(k_al2o3 * w * L)

  r_al     = ln(8*d_al    /(pi*w ))/(4*L*k_al    )
  r_quartz = ln(8*d_quartz/(pi*w2))/(4*L*k_quartz)

  rs = [r_gst,r_al,r_al2o3,r_quartz]
  rs = np.array(rs)

  R = sum(rs)

  q = To/R

  # t_layer := temperature at top surface of layer

  t_al2o3  = To      - q*r_gst
  t_al     = t_al2o3 - q*r_al2o3
  t_quartz = t_al    - q*r_al

  print 'Temperature Profile:'
  print 'GST:', To
  print 'Al2O3:', t_al2o3
  print 'Al:', t_al
  print 'Quartz:', t_quartz
  # print q*r_quartz
  print 
  print rs

def FinCapacitance():
  
  # Geometry

  w   = 500e-9
  h   = 200e-9
  dal2o3 = 20e-9
  dox = 200e-9
  dcu = 100e-9


  # Copper

  pcu = 8960 #kg/m^3
  ccu = 390  #J/kgK
  kcu = 300  #W/mK

  # SiO2

  pox = 2650
  cox = 840
  kox = 1

  # GST

  pgst = 6000
  cgst = 216
  kgst = 0.4

  # Al2O3

  kal2o3 = 10

  # Healing Length

  L = sqrt(dox*dcu*kcu/kox)

  # Volumes

  vgst = w*h
  vcu = dcu*2*(L+w/2.)
  vox = dox*2*(L+w/2.)

  # Capacitance

  Cgst = pgst*vgst*cgst
  Cox = pox*vox*cox
  Ccu = pcu*vcu*ccu

  # Resistance

  Rgst = h/(kgst*w)
  Ral2o3 = dal2o3/(kal2o3*w)
  Rcu = L/(kcu*dcu)
  Rox = dox/(kox*2*(w/2.+L))

  print L
  print 'Capacitance'
  print Cgst, Cox, Ccu, Cox/40.
  print 'Resistance'
  print Rgst, Rox, Rcu, Ral2o3

def HeaterCurrent(k=100,h=100e-9,w=2e-6,T=700,p=1.68e-8,ds=5e-4):
  print 'R/um:', p*1e-6/(w*h)
  return sqrt(4*k*h*w*T/(p*ln(8*ds/(pi*w))))


def main():
  # simpleTemp()
  # tempPlot()
  # SimpleResistor()
  # FinCapacitance()
  print HeaterCurrent(w=1e-6)
  return 0

if __name__ == '__main__':
  main()

