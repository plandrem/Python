#!/usr/bin/env python

from __future__ import division

import time
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd
import pickle as pk

from scipy.special import jn, kn, jv, kv, jvp, kvp, hankel1 as h1, h1vp
from scipy.optimize import minimize

from matplotlib.patches import Circle

import putil
import sys
import os

from scipy import pi,cos,sin,exp,sqrt
csqrt=sp.emath.sqrt

# Globals
wl = 1.
c  = 1.
k  = 2*pi/wl
wo = c*k


def getphi(sym):
    if sym == 1:   # odd  mode w.r.t x-axis
        phi = 0
        
    elif sym == 0: # even mode w.r.t x-axis
        phi = pi/2.
        
    return phi
    
def getns(N,sym,pol,harm):
    n = np.arange(N)
    
    if harm==1: return 2*n+1
    else:
        if (sym==0 and pol=='TE') or (sym==1 and pol=='TM'):
            return 2*n[1:]
        else: return 2*n

def export_detQ(dq,fname=None):
    # export |Q| to csv file
    
    if fname != None:
        sname = fname + '_dq'
    else:
        sname = 'dq'
        
    np.savetxt(sname + '_real.csv',dq.real,delimiter=',')
    np.savetxt(sname + '_imag.csv',dq.imag,delimiter=',')
    
def load_detQ(fname):
    
    print "Loading: " + fname + '...'
    d = pk.load(open(fname,'r'))
    
    return d

def getQ_complex(w,dims,nr,sym=1,N=5,harm=1,pol='TE',scaling=True, units='norm'):
    
    '''
    Builds Q-matrix representing system of linear equations for matching fields along the boundary
    of a rectangular resonator.  Based off of Goell 1969.
    
    Inputs:
    w    - complex frequency (exp[-i*(wr - i*wi)*t)
    dims - dimensions of resonator (width,height) normalized by units of high-index wavelength
    nr   - refractive index of resonator
    sym  - Consider even (0) or odd (1) symmetry across x-axis
    N    - number of harmonics to consider
    harm - {0,1,'both') determines if even (0), odd (1) or both harmonics are considered in the cylindrical harmonic expansion
    pol  - ('TE', 'TM,' or None) transverse wrt long axis of wire (same as FDFD)
    '''
    
    if units=='norm':
        e0 = 1.
        mu = 1.
        Zo = 1.
        wl = 1.
        c0 = 1.
        
    else:
        e0 = 8.85e-12
        mu = 4*pi*1e-7
        Zo = csqrt(mu/e0)
        wl = 1e-6
        c0 = 3e8
        
    
    er = nr**2
    kz = 0
    h = k1 = w / c0 * nr
    p = ko = w / c0
    
    a,b = np.array(dims) * wl/nr    # wavevectors and dimensions are now in absolute units
    if a == b: a *= 1.01            # avoid errors associated with selecting the corner point as a matching point
        
    phi = getphi(sym)
    
    # Instantize matrices
    
    eLA = np.zeros((N,N),dtype=complex) # N: number of harmonics we are using
    eLC = np.zeros((N,N),dtype=complex)
    hLB = np.zeros((N,N),dtype=complex)
    hLD = np.zeros((N,N),dtype=complex)
    eTA = np.zeros((N,N),dtype=complex)
    eTB = np.zeros((N,N),dtype=complex)
    eTC = np.zeros((N,N),dtype=complex)
    eTD = np.zeros((N,N),dtype=complex)
    hTA = np.zeros((N,N),dtype=complex)
    hTB = np.zeros((N,N),dtype=complex)
    hTC = np.zeros((N,N),dtype=complex)
    hTD = np.zeros((N,N),dtype=complex)

    # Choose angles for boundary matching conditions
    
    m = np.arange(N) + 1      # m is 1 to N
    theta = (m-0.5)*pi/(2*N)  # theta_m

    # Formulate Matrix Elements
    
    tc = sp.arctan(b/a)

    R = sin(theta)         * (theta < tc) + cos(theta + pi/4.)  * (theta==tc) + -1*cos(theta)     * (theta > tc)
    T = cos(theta)         * (theta < tc) + cos(theta - pi/4.)  * (theta==tc) +    sin(theta)     * (theta > tc)
    rm = a/(2.*cos(theta)) * (theta < tc) + (a**2+b**2)**0.5/2. * (theta==tc) + b/(2.*sin(theta)) * (theta > tc)
    
    for ni in range(N): # array (0 to N-1)
        
        # angles used for boundary matching fields at boundary depend on whether current harmonic is odd/even 
        
        # use exclusively even or odd harmonics
        if harm==1: n=2*ni+1
        elif harm==0: n=2*ni
        else: n=ni
        
        S = sin(n*theta + phi)
        C = cos(n*theta + phi)
        
        J = jn(n,h*rm)
        Jp = jvp(n,h*rm)
        JJ = n*J/(h**2*rm)
        JJp = Jp/(h)
        
        H = h1(n,p*rm)
        Hp = h1vp(n,p*rm)
        HH = n*H/(p**2*rm)
        HHp = Hp/(p)

        # scaling to prevent overflow/underflow
        
        if scaling:
            d = (a+b)/2.
            Jmult = h**2*d/abs(jn(n,h*np.amin(a,b)/2.))
            Hmult = p**2*d/abs(h1(n,p*np.amin(a,b)/2.))
        else:
            Jmult = Hmult = 1.
        
        eLA[:,ni] = J*S                                          * Jmult
        eLC[:,ni] = H*S                                          * Hmult
        hLB[:,ni] = J*C                                          * Jmult
        hLD[:,ni] = H*C                                          * Hmult
        eTA[:,ni] = 0 #-1*kz*(JJp*S*R + JJ*C*T)                  * Jmult
        eTB[:,ni] = ko*Zo*(JJ*S*R + JJp*C*T)                     * Jmult
        eTC[:,ni] = 0 #kz*(HHp*S*R + HH*C*T)                     * Hmult
        eTD[:,ni] = -1*ko*Zo*(HH*S*R + HHp*C*T)                  * Hmult
        hTA[:,ni] = er*ko*(JJ*C*R - JJp*S*T)/Zo                  * Jmult                 # typo in paper - entered as JJp rather than Jp
        hTB[:,ni] = 0 #-kz*(JJp*C*R - JJ*S*T)                    * Jmult
        hTC[:,ni] = -ko*(HH*C*R - HHp*S*T)/Zo                    * Hmult
        hTD[:,ni] = 0 #kz*(HHp*C*R - HH*S*T)                     * Hmult
        
        
        if scaling:
            eLA[:,ni] /= np.amax(abs(eLA[:,ni]))
            eLC[:,ni] /= np.amax(abs(eLC[:,ni]))
            hLB[:,ni] /= np.amax(abs(hLB[:,ni]))
            hLD[:,ni] /= np.amax(abs(hLD[:,ni]))
            eTA[:,ni] /= np.amax(abs(eTA[:,ni]))
            eTB[:,ni] /= np.amax(abs(eTB[:,ni]))
            eTC[:,ni] /= np.amax(abs(eTC[:,ni]))
            eTD[:,ni] /= np.amax(abs(eTD[:,ni]))
            hTA[:,ni] /= np.amax(abs(hTA[:,ni]))
            hTB[:,ni] /= np.amax(abs(hTB[:,ni]))
            hTC[:,ni] /= np.amax(abs(hTC[:,ni]))
            hTD[:,ni] /= np.amax(abs(hTD[:,ni]))
        
        '''
        print 'n:',n
        print 'Jmult:',Jmult
        print 'Hmult:',Hmult
        print 'abs(h1):',abs(h1(n,p*rm))
        print 'abs(h1vp):',abs(h1vp(n,p*rm))
        print 'eLA:',eLA[:,ni]
        print 'eLC:',eLC[:,ni]
        print
        '''
    O = np.zeros(np.shape(eLA))

    if pol=='TM':
        Q1 = np.hstack((eLA,-1*eLC))
        Q2 = np.hstack((hTA,-1*hTC))
        
    elif pol=='TE':
        Q1 = np.hstack((hLB,-1*hLD))
        Q2 = np.hstack((eTB,-1*eTD))
    
    if pol != None:
        Q = np.vstack((Q1,Q2))
    
    else:
        Q1 = np.hstack((eLA,O,-1*eLC,O))
        Q2 = np.hstack((O,hLB,O,-1*hLD))
        Q3 = np.hstack((eTA,eTB,-1*eTC,-1*eTD))
        Q4 = np.hstack((hTA,hTB,-1*hTC,-1*hTD))
    
        Q = np.vstack((Q1,Q2,Q3,Q4))
    
    
    # for even harmonics, eliminate n=0 terms for E or H, depending on symmetry.  Inclusion of these terms results in zero columns and thus a zero determinant.
    # Since we are eliminating columns from our matrix, we must eliminate rows as well to maintain square dimensions (4N-2).  Goell's
    # convention is to discard the first and last rows for whichever longitudinal component has odd symmetry (eg. hLB/D if sym=0)
    if pol==None:
        if harm==0:
            if sym==0:                     #eliminate b0,d0 terms
                Q = np.delete(Q,[N,3*N],1) #delete syntax: (array,index,axis (0 = row, 1 = column)
                Q = np.delete(Q,[N,2*N-1],0)
                
            elif sym==1:                   #eliminate a0,c0 terms
                Q = np.delete(Q,[0,2*N],1)
                Q = np.delete(Q,[0,N-1],0)
    
    else: # TE or TM
        if harm==0:
            if sym==0 and pol=='TE':                     #eliminate b0,d0 terms
                Q = np.delete(Q,[0,N],1) 
                Q = np.delete(Q,[0,N-1],0)
                
            elif sym==1 and pol=='TM':                   #eliminate a0,c0 terms
                Q = np.delete(Q,[0,N],1) 
                Q = np.delete(Q,[0,N-1],0)
        
    return Q
    
def get_coefs(w,dims,nr,sym,N,harm,pol):
    '''
    Returns A,C or B,D depending on polarization.  Coefficients obtained from the nullspace of Q, 
    ie the solution of the matrix equation Q*T=0.
    '''
    
    q = getQ_complex(w,dims,nr,sym,N,harm,pol,scaling=False,units='norm')
    u,s,v = np.linalg.svd(q)
    coefs = v[:,-1]
    C1,C2 = np.split(coefs,2)
    
    return C1,C2
    
    
def get_fields(w,dims,nr,sym,N,harm,pol,res=200,ax=None):
    '''
    Returns the complex field values at specified points in the 2D cross section of a rectangular resonator
    
    INPUTS
    w       - complex freq. using exp[-iwt] convention (wr - 1j*wi)
    dims    - dimensions of resonator normalized to wavelength
    nr      - refractive index of resonator
    sym     - symmetry of dominant field (z component) w.r.t. x axis
    N       - Number of cylindrical harmonics considered in series
    harm    - (0,1) mode generated by even or odd harmonics (determines y-symmetry)
    pol     - ('TE','TM') polarization; TE implies Hz
    res     - resolution of field data
    ax      - axes on which to plot fields; if None, generates new figure
    
    OUTPUTS
    Fz, Fx, Fy - complex field data for each component.  If Fz is Hz, then Fx,y will be Ex,y and vice-versa.
    '''
    
    a,b = dims
    
    # grid dimensions in normalized length units
    xmax = a + 1
    ymax = b + 1
    
    xs = np.linspace(-xmax,xmax,res)
    ys = np.linspace(-ymax,ymax,res)
    
    X,Y = np.meshgrid(xs,ys)
    
    R = sqrt(X**2 + Y**2)
    TH = sp.arctan(Y/X)
    
    ko = w / c
    k1 = ko * nr
    
    ns = getns(N,sym,pol,harm)
    phi = getphi(sym)
    
    mask = (abs(X) <= a/2.) * (abs(Y) <= b/2.)
    
    C1,C2 = get_coefs(w,dims,nr,sym,N,harm,pol)
    
    if pol=='TE': #Fz = Hz
        Fz_int = np.sum([C1[ni]*jn(n,k1*R)*cos(n*TH + phi) for ni,n in enumerate(ns)],axis=0)
        Fz_ext = np.sum([C2[ni]*h1(n,ko*R)*cos(n*TH + phi) for ni,n in enumerate(ns)],axis=0)
    
    if pol=='TM': #Fz = Ez
        Fz_int = np.sum([C1[ni]*jn(n,k1*R)*sin(n*TH + phi) for ni,n in enumerate(ns)],axis=0)
        Fz_ext = np.sum([C2[ni]*h1(n,ko*R)*sin(n*TH + phi) for ni,n in enumerate(ns)],axis=0)
        
    Fz_tot = Fz_int * mask + Fz_ext * (~mask)

    ext = [-xmax,xmax,-ymax,ymax]
    
    # create shaded box to display resonator
    rect = plt.Rectangle((-a/2.,-b/2.),a,b,facecolor='k',alpha=0.3)
    
    if ax==None:
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        
    ax.imshow(abs(Fz_tot),interpolation='nearest',extent=ext, vmax = np.amax(abs(Fz_int*mask)))   
    #ax.add_patch(rect)
    
def detQ_complex(w,dims,nr,sym=1,N=5,harm=1,pol='TE'):
    Q = getQ_complex(w,dims,nr,sym,N,harm,pol)
    return np.linalg.det(Q)
    
def logabsdq(w,dims,nr,sym=1,N=5,harm=1,pol='TE'):
    Q = getQ_complex(w,dims,nr,sym,N,harm,pol)
    dq = np.linalg.det(Q)
    return np.log10(abs(dq))
    
def map_detQ_w(dims=(2,1),nr=4,sym=0,harm=0,pol='TE',N=5,res=100,save=True):
    '''
    Generate a map of |Q| over some range of complex frequency space
    '''
    
    wr = np.linspace(1e-12,2,res) * wo
    wi = np.linspace(-1.5,1.5,res) * wo
    
    w = [x - 1j*y for x in wr for y in wi]
    w = np.array(w)
    
    print 'Generating Determinant Values...'
    
    dq = np.array(putil.maplist(detQ_complex,w,dims,nr,sym,N,harm,pol))
    dq = dq.reshape(len(wr),len(wi)).transpose()

    print '...done.'
    print
    
    d = {
        'dq':dq,
        'wr':wr,
        'wi':wi,
        'dims':dims,
        'nr':nr,
        'sym':sym,
        'harm':harm,
        'N':N,
        'pol':pol
        }
    
    if save:
        sn = get_sname(d)
        pk.dump(d, open(sn + '.pk','w'))
        
        print 'Simulation data saved as "' + sn + '.pk"'
    
    return dq
    
def get_sname(d):
    
    sname = ''
    
    for key in d:
        if key in ['N','nr','sym','harm','pol']: sname += key + '.' + str(d[key]) + '_'
        elif key == 'dims': sname += 'a.' + str(d['dims'][0]) + '_b.' + str(d['dims'][1]) + '_'
        
    sname = sname[:-1]
    
    return sname

def plot_detQ(d,ax=None):
    '''
    Plots map of |Q| using dictionary of simulation parameters, d.  If ax is provided, plots on those axes.
    '''
    
    wr = d['wr']
    wi = d['wi']
    dq = d['dq']
    
    ext = np.array([np.amin(wr),np.amax(wr),np.amin(wi),np.amax(wi)]) / wo
    
    vmin = None
    vmax = None

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    im = ax.imshow(sp.log10(abs(dq)),origin='lower',interpolation='nearest',extent=ext,vmin=vmin, vmax=vmax,aspect='equal')
    ax.figure.colorbar(im,ax=ax)
    
    return im
    
class DQmap:
    
    def __init__(self, d):


        self.args = [
            d['dims'],
            d['nr'],
            d['sym'],
            d['N'],
            d['harm'],
            d['pol']
        ]
        
        #Class Variables
        self.ctrl  = False
        self.shift = False
        
        #Plot objects
        self.map_fig = plt.figure()
        self.map_ax = self.map_fig.add_subplot(111)
        self.im = plot_detQ(d,ax=self.map_ax)
        
        self.field_fig = plt.figure()
        self.field_ax = self.field_fig.add_subplot(111)
        
        #Event Handlers
        self.clickevent = self.map_fig.canvas.mpl_connect('button_press_event', self.click)
        self.keydownevent = self.map_fig.canvas.mpl_connect('key_press_event', self.keydown)
        self.keyupevent = self.map_fig.canvas.mpl_connect('key_release_event', self.keyup)
        
    def click(self,event):
        
        if self.ctrl==True or self.shift==True:
            w = event.xdata - 1j * event.ydata
            if w.imag >= 0: sign = '+'
            else: sign='-'
            
            print "Frequency: %.2f " % w.real + sign + " %.2fi" % abs(w.imag)
            print '|Q|: %.2f' % sp.log10(abs(detQ_complex(w*wo,*self.args)))
            print
            
            if self.shift==True: w1 = self.find_min(w)
            else: w1 = w
        
            self.update_field_plot(w1)
            self.draw_circle(w1)
            
        
    def keydown(self,event):
        if event.key=='control': self.ctrl=True
        if event.key=='shift'  : self.shift=True

    def keyup(self,event):
        if event.key=='control': self.ctrl=False
        if event.key=='shift'  : self.shift=False
            
    def update_field_plot(self,w):
        
        try:
            self.field_fig
        except:
            self.field_fig = plt.figure()
            self.field_ax = self.field_fig.add_subplot(111)
            
        get_fields(w,*self.args,res=200,ax=self.field_ax)
        #self.field_im = self.field_ax.imshow(Fz)
        self.field_fig.canvas.draw()
        
    def find_min(self,w):
        
        w_min = minimize(logabsdq,w*wo,args=args)
        w_min /= wo
        
        print w_min
        
        return w_min
        
    def draw_circle(self,w):
        
        self.map_ax.patches = []
        
        xy = (w.real,-w.imag)
        circle = Circle(xy,radius=0.1,edgecolor='g',facecolor='none',lw=2)
        
        self.map_ax.add_patch(circle)
        self.map_fig.canvas.draw()
        
def test_load():
    
    f1 = sys.argv[2]
    f2 = sys.argv[3]
    
    d1 = load_detQ(f1)
    d2 = load_detQ(f2)
    
    fig, ax = plt.subplots(1,2)
    
    ax[0].imshow(np.log10(abs(d1['dq'])),interpolation='nearest')
    ax[1].imshow(np.log10(abs(d2['dq'])),interpolation='nearest')
    
def test_mapdq():
    
    fig, ax = plt.subplots(2,2)
    
    for i in range(2):
        for j in range(2):
        
            dq = map_detQ_w(dims=(2,1),nr=4,sym=i,harm=j,pol='TE',N=20,res=200,save=True)
            
            ax[i,j].imshow(np.log10(abs(dq)),interpolation='nearest')
            
                

if __name__ == '__main__':
    
    if   '-tl' in sys.argv: test_load()    
    elif '-tdq' in sys.argv: test_mapdq()    
    
    elif '-l' in sys.argv:    # load existing data
        i = sys.argv.index('-l'); ln = sys.argv[i+1]
        d = load_detQ(fname=ln)
    
        m = DQmap(d)
        
    else:
        args = {
            'res'     : 200,
            'dims'    : (2,1),
            'nr'      : 4.,
            'sym'     : 1,
            'N'       : 20,
            'harm'    : 1,
            'pol'     : 'TE'
        }
        
        map_detQ_w(**args)
    
    plt.show()