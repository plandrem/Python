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
    a = do**3*3/(4*pi) * (eps-1)/(eps+2)

    self.a = a

  


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

  def getQ_periodic(self,wl):
    '''
    '''

    k = 2*pi/wl
    
    dx,dy = distanceMatrices(self.x,self.y)

    # R = sqrt(dx**2 + dy**2)

    g = 1e-2
    rmax = 16**(0.25)/(g*k)
    M = int(rmax/self.do) # number of periodic units such that Blaine's arbitrary scaling factor drops to e^-16

    A = np.eye(self.N) * 1/self.a
    Q = A

    for m in range(-M,M):

      print 'Solving unit cell %u of %u' % (m,M)

      dz = m*self.do * np.ones(np.shape(dx))

      R = np.array([dx,dy,dz])
      E = dipoleRadiation(wl,1,Rm) * exp(-1*(g*k*Rm)**4)
      E = np.nan_to_num(E)

      Q -= E

    return Q

  def getQ_crude(self,wl):
    '''
    generates Q matrix using for loops instead of matrices.  This method was introduced to circumvent
    potential bugs hidden in abstract array methods.
    '''

    k = 2*pi/wl

    a = self.a
    a = a/(1-2j/3*k**3*a) #radiation reaction (BT Draine, 1988)

    N = self.N

    Q = np.zeros((N,N),dtype=complex)

    for i in range(N):
      for j in range(N):

        if i == j:
          Q[i,j] = 1/self.a

        else:
          # get coordinates of the two dipoles
          xi = self.x[i]
          xj = self.x[j]
          yi = self.y[i]
          yj = self.y[j]

          # get distance between dipoles
          r = sqrt((xi-xj)**2 + (yi-yj)**2)

          Q[i,j] = -exp(1j*k*r) / r**3 * (k**2 * r**2 + 1j*k*r - 1)

    return Q


  def getQ_crude_periodic(self,wl):
    '''
    '''

    k = 2*pi/wl

    a = self.a
    a = a/(1-2j/3*k**3*a) #radiation reaction (BT Draine, 1988)

    N = self.N

    Q = np.zeros((3*N,3*N),dtype=complex)

    g = 1e-2
    rmax = 16**(0.25)/(g*k)
    M = int(rmax/self.do) # number of periodic units such that Blaine's arbitrary scaling factor drops to e^-16
    M=100

    for m in range(-M,M):

      print 'Solving unit cell %u of %u' % (m,M)
      for i in range(N): # dipole at which we are combining fields
        for j in range(N): # dipole which is emitting field

          # print i,j
          
          if i == j and m == 0:
            pass

          else:
            # get coordinates of the two dipoles
            xi = self.x[i]
            xj = self.x[j]
            yi = self.y[i]
            yj = self.y[j]

            # get distance between dipoles in unit cell
            rx = xj-xi
            ry = yj-yi
            rz = m*self.do
            r  = np.linalg.norm([rx,ry,rz])

            C1 = lambda r1,r2,r3: -(k**2) * (r2**2+r3**2) + (1-1j*k*r)/r**2 * (r**2 - 3*r1**2) * exp(1j*k*r)/r**3 * exp(1j*k*m*self.do - (g*k*r)**4)
            C2 = lambda r1,r2:     (k**2) * (r1*r2)       + (1-1j*k*r)/r**2 * (r**2 - 3*r1*r2) * exp(1j*k*r)/r**3 * exp(1j*k*m*self.do - (g*k*r)**4)

            Q[3*i+0,3*j+0] += C1(rx,ry,rz)
            Q[3*i+0,3*j+1] += C2(rx,ry)
            Q[3*i+0,3*j+2] += C2(rx,rz)
            Q[3*i+1,3*j+0] += C2(ry,rx)
            Q[3*i+1,3*j+1] += C1(ry,rz,rx)
            Q[3*i+1,3*j+2] += C2(ry,rz)
            Q[3*i+2,3*j+0] += C2(rz,rx)
            Q[3*i+2,3*j+1] += C2(rz,ry)
            Q[3*i+2,3*j+2] += C1(rz,rx,ry)

    Q += np.eye(3*N) / a

    return Q

  def exciteByPW(self,wl,angle):
    x = self.x
    y = self.y

    Q = self.getQ_crude_periodic(wl)

    # Q = self.getQ(wl)
    # Qcrude = self.getQ_crude(wl)

    # print np.amax(abs(Q-Qcrude))
    # exit()

    E = PW_polarized(x,y,angle,wl,'Ez')


    # n = sqrt(self.N)
    # Esq = E.reshape(n,n)
    # plt.imshow(Esq.real)
    # plt.show()
    # exit()

    # solve equation Q*P = E for P, the polarization states of the dipoles

    self.wl = wl
    self.angle = angle
    self.P = np.linalg.solve(Q,E).reshape(self.N,3)
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
    p = self.P[:,2].real

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

  return exp(1j*k*r) / r**3 * (-k**2 * r**2 + 1j*k*r - 1) * P

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

  vmax = 1
  vmin = -vmax
  plt.imshow(Fz.real,extent=putil.getExtent(x,y),cmap='RdBu',vmax=vmax,vmin=vmin)
  plt.colorbar()
  plt.show()

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
    axis=0

  
  k = 2*pi/wl
  d = np.array([np.linalg.norm(i) for i in r])
  d = np.vstack([d,d,d]).T

  rxP = np.cross(r,P)
  rxrxP = np.cross(r,rxP)

  rdotP = np.sum(r*P,axis=axis)
  rdotP = rdotP.reshape(len(rdotP),1)

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

def blockMatrix(x):
  '''
  given an Nx x Ny x 3 array, returns a 3Nx x 3Ny array where each 3-vector has been converted into a diagonal block
  '''
  print x.shape
  nx,ny, = np.shape(x)
  M = np.zeros((3*nx,3*ny),dtype=complex)

  for i in range(nx):
    for j in range(ny):

      M[3*i+0,3*j+0] = x[i,j,0]
      M[3*i+1,3*j+1] = x[i,j,1]
      M[3*i+2,3*j+2] = x[i,j,2]

  return M

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

    dx = xi - xj
    dy = yi - yj

    return dx,dy

      
def test_distanceMatrices():

  rect = Rect(w,h)

  x = rect.x
  y = rect.y

  dx,dy = distanceMatrices(x,y)

  plt.figure()
  plt.imshow(dx,origin='upper')
  plt.colorbar()
  plt.title('x-distance between ith and jth dipoles')
  plt.xlabel('i')
  plt.ylabel('j')

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

  angle = 25.
  wl = 2.

  x = np.linspace(-10,10,1000)
  y = np.linspace(-10,10,1000)

  X,Y = np.meshgrid(x,y)

  pw = PW(X,Y,angle,wl)

  plt.imshow(pw.real,extent=putil.getExtent(x,y),cmap='RdBu')
  plt.colorbar()
  plt.show()
  
def PW_polarized(x,y,angle,wl,pol='Ez'):
  '''
  returns the complex amplitude of a plane wave at point x,y, traveling at an angle.  The phase of 
  the PW is zero at the point 0,0.  Angle is in degrees, with 0 representing normal incidence from above.

  Polarization: Ez or Hz

  Returns 3-vector with components of E field at given point
  '''

  alpha = angle * pi/180.
  k = 2*pi/wl
  phi = 1j*(x*k*sin(alpha) - y*k*cos(alpha))

  if pol in ['Ez','TM']:
    E = np.array([[0,0,exp(p)] for p in phi])

  elif pol in ['Hz','TE']:
    E = np.array([cos(alpha),sin(alpha),0]) * exp(phi)

  return E.flatten()
  
  

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

  width = 99.
  height = 99.

  rect = Rect(width,height,eps=16,do=15)

  x = rect.x
  y = rect.y

  wl = 400.
  angle = 45.

  rect.exciteByPW(wl,angle)

  # print rect.getQabs()
  
  # rect.plotField(-2*width,2*width,-2*height,2*height)
  # rect.plotFF()
  # rect.debug()
  rect.plotDipoles()

  plt.show()


  return 0

if __name__ == '__main__':
  main()
  # mode_solver()
  # example_scatteredField()
  # test_dipoleRadiation3D()
  # test_Rect_discretize()
  # test_distanceMatrices()
  # test_PW()
