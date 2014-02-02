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

'''
Global Params
'''


class Rect():

  def __init__(self,w,h,eps=1,do=5e-2):
    '''
    w - width
    h - height
    eps - permittivity
    do  - lattice spacing of resonator dipoles
    '''

    self.do = do

    self.width = w
    self.height = h

    self.x, self.y = self.discretize()

    self.N = len(self.x) # number of dipoles

    N = 1/do**3
    # self.a = (1 - eps) / (-4*pi/3*N*(eps + 1) - 4*pi*N)
    self.a = 3*do**3/(4*pi) * (eps-1)/(eps+2)

  


  def discretize(self):
    '''
    returns lists of x and y coordinates of discrete dipoles
    '''

    xs = np.arange(0.,self.width,self.do)
    ys = np.arange(0.,self.height,self.do)

    xs -= np.amax(xs/2.)
    ys -= np.amax(ys/2.)

    pairs = np.array([[i,j] for i in xs for j in ys])

    xlist = pairs[:,0]
    ylist = pairs[:,1]

    return xlist,ylist

  def getQ(self,wl):
    '''
    Generates Q matrix for the resonator at a given wavelength.  The Q matrix 
    represents the interaction between all dipoles, hence the determinant of Q
    should go to zero in the presence of natural modes
    '''
    
    dx,dy = distanceMatrices(self.x,self.y)

    R = sqrt(dx**2 + dy**2)

    A = np.eye(self.N) * 1/self.a

    E = dipoleRadiation(wl,1,R)
    E = np.nan_to_num(E)

    Q = A - E

    return Q

  def exciteByPW(self,wl,angle):
    x = self.x
    y = self.y

    Q = self.getQ(wl)
    E = PW(x,y,angle,wl)

    # solve equation Q*P = E for P, the polarization states of the dipoles

    self.wl = wl
    self.angle = angle
    self.P = np.linalg.solve(Q,E)
    self.Q = Q
    self.E = E

  def plotField(self,xmin,xmax,ymin,ymax,xres=100,yres=100):

    xs = np.linspace(xmin,xmax,xres)
    ys = np.linspace(ymin,ymax,yres)

    X,Y = np.meshgrid(xs,ys)
    Fz = np.zeros((len(xs),len(ys)),dtype=complex)

    # loop over all dipoles and sum contribution to scattered field
    for n in range(self.N):
      print 'plotField: processing dipole %u of %u' % (n,self.N)
      i = self.x[n]
      j = self.y[n]

      r = sqrt((X-i)**2 + (Y-j)**2)

      Fz += dipoleRadiation(self.wl,self.P[n],r)


    Fz_crop = Fz * ((abs(X) > self.width) + (abs(Y) > self.height) == 1) # field with resonator removed
    vmax = np.amax(abs(Fz_crop)**2)
    # vmax = None

    plt.figure()
    # plt.imshow(Fz.real,cmap='RdBu',vmax=vmax,vmin=-vmax)
    plt.imshow(abs(Fz)**2,extent=putil.getExtent(xs,ys),cmap='hot',vmax=vmax,vmin=0)
    # plt.imshow(Fz.real,extent=putil.getExtent(xs,ys),cmap='RdBu',vmax=vmax,vmin=-vmax)
    plt.colorbar()

  def plotFF(self,res=100,R=1e9,vmax=None):
    '''
    
    '''

    theta = np.linspace(0,2*pi,res)

    xs = R*cos(theta)
    ys = R*sin(theta)

    Fz = np.zeros(len(theta),dtype=complex)

    # loop over all dipoles and sum contribution to scattered field
    for n in range(self.N):
      print 'plotFF: processing dipole %u of %u' % (n,self.N)
      i = self.x[n]
      j = self.y[n]

      r = sqrt((xs-i)**2 + (ys-j)**2)

      Fz += dipoleRadiation(self.wl,self.P[n],r)

    plt.figure()
    plt.subplot(111,polar=True)
    plt.gca().grid(True)

    plt.plot(theta,abs(Fz)**2,color='r',lw=2)


  def plotDipoles(self):
    '''
    shows solution to matrix equation
    '''
  
    plt.figure()

    x = self.x
    y = self.y
    p = self.P.real

    vmax = abs(np.amax(p))
    vmin = -vmax

    plt.scatter(x,y,s=100,c=p,cmap='RdBu',vmax=vmax,vmin=vmin)
    plt.colorbar()
    
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_axis_bgcolor('k')

    plt.xlim(-0.6*self.width,0.6*self.width)
    plt.ylim(-0.6*self.height,0.6*self.height)

    plt.title('Dipole Polarization')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')

  def getQabs(self):
    '''
    using eps = mu = 1 convention
    '''
    k = 2*pi/self.wl
    w = k

    a = self.a

    pwrAbs_n = -0.5*w*a.imag * (abs(self.P)**2) / abs(a)**2         # power absorbed by nth dipole
    pwrAbs = sum(pwrAbs_n)

    Eo = 1
    Z = 1
    crossSection = (self.width * cos(self.angle) + self.height * sin(self.angle))

    pwrInc = 1/2. * abs(Eo)**2/Z * crossSection

    Qabs = pwrAbs/pwrInc

    return Qabs
    
  def debug(self):
    print "dipole locations and polarization:"
    for n in range(self.N):
      print "(%.3e, %.3e)  %.3e + %.3ej" % (self.x[n],self.y[n],self.P[n].real,self.P[n].imag)

    print
    print "Polarizabiltiy:",self.a
    print
    print "Q:",self.Q
    print
    print "E:",self.E
    print 
    print np.amax(abs(self.E-np.dot(self.Q,self.P)))


  


    

def test_Rect_discretize():
  '''
  
  '''

  r = Rect(1,0.5)

  for i in range(r.N):
    plt.plot(r.x[i],r.y[i],'ro')

  plt.gca().set_aspect('equal')
  plt.show()
  



def dipoleRadiation(wl,P,r):
  '''
  returns complex field Fz at a distance r from a source dipole with complex moment P
  '''

  k = 2*pi/wl

  return exp(1j*k*r) / r**3 * (k**2 * r**2 - (1 - 1j*k*r)) * P

def test_dipoleRadiation():
  '''
  
  '''

  wl = 2
  P = 1

  x = np.linspace(-5,5,100)
  y = np.linspace(-5,5,100)

  X,Y = np.meshgrid(y,x)

  r = sqrt(X**2 + Y**2)

  Fz = dipoleRadiation(wl,P,r)

  plt.imshow(Fz.real,extent=putil.getExtent(x,y),cmap='RdBu')
  plt.show()

def getSquareArray(x):
  '''
  repeats a 1D array to produce a square matrix
  '''

  return np.repeat(np.vstack([x]),len(x),axis=0)
  
def distanceMatrices(x,y):
    '''
    returns NxN matrices dx,dy containing the x and y coordinate distance between dipoles i and j
    '''
  
    xi = getSquareArray(x)
    yi = getSquareArray(y)

    xj = np.transpose(xi)
    yj = np.transpose(yi)

    dx = xj - xi
    dy = yj - yi

    return dx,dy

      
def test_distanceMatrices():

  rect = Rect(1.,1.,do=0.2)

  x = rect.x
  y = rect.y

  dx,dy = distanceMatrices(x,y)

  plt.figure()
  plt.imshow(dx,origin='upper')
  plt.colorbar()
  plt.title('x-distance between ith and jth dipoles')
  plt.xlabel('i')
  plt.ylabel('j')

  print x
  print y

  plt.show()

def PW(x,y,angle,wl):
  '''
  returns the complex amplitude of a plane wave at point x,y, traveling at an angle.  The phase of 
  the PW is zero at the point 0,0.  Angle is in degrees, with 0 representing normal incidence from above.
  '''

  alpha = angle * pi/180.
  k = 2*pi/wl
  phi = 1j*(x*k*sin(alpha) - y*k*cos(alpha))

  return exp(phi)

def test_PW():
  '''
  
  '''

  angle = 45.
  wl = 1.

  x = np.linspace(-5,5,500)
  y = np.linspace(-5,5,500)

  X,Y = np.meshgrid(x,y)

  pw = PW(X,Y,angle,wl)

  plt.imshow(pw.real,extent=putil.getExtent(x,y),cmap='RdBu')
  plt.colorbar()
  plt.show()
  
  
  

def mode_solver():

  '''
  doesn't work :/
  '''

  rect = Rect(1,0.5,eps=16)

  x = rect.x
  y = rect.y

  wls = np.linspace(0.3,1,100)

  dq = []

  # wl = 0.9

  # q = rect.getQ(wl)

  # q /= np.amax(abs(q))

  # print np.linalg.det(q)

  # plt.imshow(q.real,origin='upper')
  # plt.colorbar()

  for wl in wls:
    q  = rect.getQ(wl)
    q /= np.amax(abs(q))
    dq.append(np.linalg.det(q))

  dq = np.array(dq)

  plt.plot(wls,sp.log(abs(dq)),lw=2)
  # plt.ylim(-1.2,1.2)

  plt.show()


  return 0

def example_scatteredField():

  rect = Rect(1,0.5,eps=16,do=5e-2)

  x = rect.x
  y = rect.y

  print x[0],y[0]

  wl = 1.
  angle = 0.

  Q = rect.getQ(wl)
  E = PW(x,y,angle,wl)

  # solve equation Q*P = E for P, the polarization states of the dipoles

  P = np.linalg.solve(Q,E)

  # print P

  xs = np.linspace(-3,3,100)
  ys = np.linspace(-3,3,100)

  X,Y = np.meshgrid(xs,ys)
  Fz = np.zeros((len(xs),len(ys)),dtype=complex)

  # loop over all dipoles and sum contribution to scattered field
  for n in range(len(x)):
    print 'processing dipole %u of %u' % (n,rect.N)
    i = x[n]
    j = y[n]

    r = sqrt((X-i)**2 + (Y-j)**2)

    Fz += dipoleRadiation(wl,P[n],r)


  vmax = 0.01

  # plt.imshow(Fz.real,cmap='RdBu',vmax=vmax,vmin=-vmax)
  plt.imshow(abs(Fz)**2,cmap='hot',vmax=vmax,vmin=0)
  # plt.imshow(Fz.real,extent=putil.getExtent(xs,ys),cmap='RdBu',vmax=vmax,vmin=-vmax)
  plt.colorbar()
  plt.show()


  return 0

def example_FF():

  rect = Rect(1,0.5,eps=16,do=5e-2)

  x = rect.x
  y = rect.y

  wl = 1.
  angle = 0.

  Q = rect.getQ(wl)
  E = PW(x,y,angle,wl)

  # solve equation Q*P = E for P, the polarization states of the dipoles

  P = np.linalg.solve(Q,E)

  # print P

  # theta=0 points along x, moving CCW
  theta = np.linspace(0,2*pi,200)
  R = 1e2
  xs = R*cos(theta)
  ys = R*sin(theta)

  Fz = np.zeros(len(xs),dtype=complex)

  # loop over all dipoles and sum contribution to scattered field
  for n in range(len(x)):
    print 'processing dipole %u of %u' % (n,rect.N)
    i = x[n]
    j = y[n]

    r = sqrt((xs-i)**2 + (ys-j)**2)

    Fz += dipoleRadiation(wl,P[n],r)

  plt.subplot(111,polar=True)
  plt.gca().grid(True)

  plt.plot(theta,abs(Fz)**2,color='r',lw=2)
  plt.show()


  return 0

def main():

  width = 1.
  height = 1.

  rect = Rect(width,height,eps=16,do=0.2)

  x = rect.x
  y = rect.y

  wl = 1.
  angle = 0.

  rect.exciteByPW(wl,angle)

  # print rect.getQabs()
  
  # rect.plotField(-2*width,2*width,-2*height,2*height)
  # rect.plotFF()
  rect.debug()
  rect.plotDipoles()

  plt.show()


  return 0

if __name__ == '__main__':
  main()
  # mode_solver()
  # example_scatteredField()
  # test_dipoleRadiation()
  # test_Rect_discretize()
  # test_distanceMatrices()
  # test_PW()
