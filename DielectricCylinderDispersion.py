#!/usr/bin/env python


from __future__ import division

from numpy import *
from scipy import *
from pylab import *
from scipy.special import jn, kn, jvp, kvp, hankel1 as h1n, h1vp
from scipy.optimize import fsolve

from matplotlib.colors import LinearSegmentedColormap

import putil


#Fundamental constants
c = 3e8									#Speed of Light (m/s)
mu = 4 * pi * 1e-7							#Permeability of Free Space (H/m)
e0 = 8.85e-12								#Permittivity of Free Space (F/m)


#def ResonanceMap(wls,rads,n,pol='TE'):

    ##Wave properties
    #f = c / wls								#Frequency (Hz)
    #w = 2*pi*f								#Angular Frequency (rad/s)
	    
    #d = 2*rads										
    #ko = 2*pi / wl								#Wavevectors (m^-1)

    #imi = zeros((len(wls),len(rads)))
    #im = zeros((len(wls),len(rads)))

    #X,Y = meshgrid(ko,rads)
    #kos=ravel(X)
    #aas=ravel(Y)

    #ns = []
    #for x in rads:
	    #ns.extend(list(n))

    #plots = range(4)
    #for j in range(len(plots)):
	    #print j

	    #if pol == 'TE': dispersion = lambda x,y,n: 1/n * jvp(j,n*x*y) / jn(j,n*x*y) - kvp(j,x*y) / kn(j,x*y)
	    #if pol == 'TM': dispersion = lambda x,y,n: n * jvp(j,n*x*y) / jn(j,n*x*y) - kvp(j,x*y) / kn(j,x*y)

	    #imi_flat = array(map(dispersion,kos,aas,ns))
	    #imi = imi_flat.reshape(len(wl),len(a))
	    #imi = flipud(imi)
	    
	    #print amin(abs(imi))
	    #im = im + (j+1)*(abs(imi)<.2)

    #xmin = amin(wls)
    #xmax = amax(wls)
    #ymin = amin(d)
    #ymax = amax(d)
    #extent = array([xmin,xmax,ymin,ymax]) * 1e9

    ## Custom Colormap -- example documentation at http://matplotlib.sourceforge.net/examples/pylab_examples/custom_cmap.html

    #cdict = {'red':   ((0.0, 0.0, 0.0),
		       #(0.1, 0.0, 1.0),
		       #(0.26, 1.0, 0.0),
		       #(0.51, 0.0, 0.0),
		       #(0.76, 0.0, 1.0),
		       #(1.0, 1.0, 0.0)),
		      
		      #'green': ((0.0, 0.0, 0.0),
		       #(0.26, 0.0, 1.0),
		       #(0.51, 1.0, 0.0),
		       #(0.76, 0.0, 0.9),
		       #(1.0, 0.9, 0.0)),

		      #'blue':   ((0.0, 0.0, 0.0),
		       #(0.26, 0.0, 0.0),
		       #(0.51, 0.0, 1.0),
		       #(0.76, 1.0, 0.0),
		       #(1.0, 0.0, 0.0))

	    #}
	    
    #cm = LinearSegmentedColormap('custom', cdict)
    #plt.register_cmap(cmap=cm)

    #imshow(abs(im),extent=extent,interpolation='nearest',cmap='custom', aspect='equal')
    #xlabel('wavelength (nm)')
    #ylabel('diameter (nm)')
    #colorbar()

    #show()

def getResonantRadii(wl,rads,n,no=1,pol='TE',jmax=4,mmax=4):

    #Wave properties
    f = c / wl		#Frequency (Hz)
    w = 2*pi*f		#Angular Frequency (rad/s)
	    
    d = 2*rads										
    ko = 2*pi / wl	#Wavevector (m^-1)
        
    # Return array of resonant radii
    Rs = zeros((jmax,mmax),dtype=float) * nan
    
    # iterate over mode order j
    for j in range(jmax):

	if pol == 'TE': dispersion = lambda x: no/n * jvp(j,n*x*ko) / jn(j,n*x*ko) - h1vp(j,x*ko*no) / h1n(j,x*ko*no)
	if pol == 'TM': dispersion = lambda x: n/no * jvp(j,n*x*ko) / jn(j,n*x*ko) - h1vp(j,x*ko*no) / h1n(j,x*ko*no)
	
	Fj = [dispersion(x) for x in rads]
		
	# All zero crossings are conveniently positive-negative with increasing radius.  There are neg-pos discontinuities which must be avoided.
	# Find sign changes:
	xing = diff(sign(Fj))
	#plot(rads[1:],xing)
	
	# retrieve radii corresponding to negative sign changes
	resonances = rads[where(xing==-2)]
	
	#truncate to maximum # of radial mode order
	resonances = resonances[:mmax]
	
	Rs[j,:len(resonances)] = resonances
	
    return Rs
    
def ResonanceMap(wls,rads,n,no=1,pol='TE',jmax=4,mmax=4):
    
    Res = zeros((jmax,mmax,len(wls)),dtype=float) * nan
    
    for i,wl in enumerate(wls):
	Res[:,:,i] = getResonantRadii(wl,rads,n[i],no,pol,jmax,mmax)
	
    cs = ['k','r','b','m','c','g'] 	#color denotes radial mode order
    ls = ['-','--',':'] 		#line style denotes azimuthal mode order
    
    fig = plt.figure()
    for m in range(mmax):
	for j in range(jmax):
	    plt.plot(wls*1e9,2*Res[j,m,:]*1e9,color=cs[j%len(cs)],ls=ls[m%len(ls)],marker='o')
	    
    xlabel('Wavelength (nm)')
    ylabel('Diameter (nm)')
    
    xlim(amin(wls)*1e9,amax(wls)*1e9)
    ylim(amin(rads)*2e9,amax(rads)*2e9)
    
    #plt.show()
    
def effectiveIndex(wl,r,j):
    return wl*j/(2*pi*r)
    
def example_effectiveIndex():
    
    wl = 1500e-9
    rads = linspace(50,1000,500)*1e-9
    
    jmax=4
    mmax=5
    
    resrad = getResonantRadii(wl,rads,4,'TE',jmax,mmax)
    
    for j in range(jmax):
	plot(resrad[j,:]*1e9,effectiveIndex(wl,resrad[j,:],j),marker='o')
	
    show()
    
def ModePlot(j,m,n,pol='TE',phi=0):
    '''
    Displays the field components for a given mode (eg. TE_jm)
    
    j: azimuthal mode order
    m: radial mode order
    n: cylinder index
    pol: ['TE','TM'] polarization (transverse relative to cylinder axis)
    phi: azimuthal phase - rotates field profile clockwise
    
    Notes: rad/wl pairs for proper modes to be obtained from ResonanceMap()
    
    '''
    
    plot_resolution = 200
    solver_resolution = 1e3
    
    wl = 600 #arbitrary
    c = 1
    mu = 1
    
    k = 2*pi*n/wl
    g = 2*pi/wl
    w = 2*pi*c/wl
    
    # find radius for selected mode
    rads = linspace(0.01*wl/n,2*wl/n,solver_resolution) #radii to consider - increase resolution for better field accuracy
    Rs = getResonantRadii(wl,rads,n,pol=pol,jmax=j+1,mmax=m+1)
    rad = Rs[j,m]
    print 2*Rs
    print 2*rad
    
    if str(rad) == 'nan':
	print "Mode not found - expand range of radii"
	exit()
    
    # get cartesian coordinates for plotting
    q = 1.5 # number of radii to include on sides of resonator
    x = y = linspace(-q*rad,q*rad,plot_resolution)
    X,Y = meshgrid(x,y)
    Y *= -1 #y grid is formed upside down
    
    # Convert cartesian coordinates to r,theta pairs
    R = sqrt(X**2 + Y**2)
    T = arctan2(Y,X)
        
    if pol == 'TE':
	Fz_in = lambda r,t: jn(j,k*r)/jn(j,k*rad)*cos(j*t+phi)
	Ft_in = lambda r,t: 1j/k**2*(-w*mu*k*jvp(j,k*r)/jn(j,k*rad)*cos(j*t+phi))
	Fr_in = lambda r,t: 1j/k**2*(w*mu/r*jn(j,k*r)/jn(j,k*rad)*j*(-sin(j*t+phi)))
	
	Fz_out = lambda r,t: h1n(j,g*r)/h1n(j,g*rad)*cos(j*t)
	Ft_out = lambda r,t: 1j/g**2*(-w*mu*g*h1vp(j,g*r)/h1n(j,g*rad)*cos(j*t+phi))
	Fr_out = lambda r,t: 1j/g**2*(w*mu/r*h1n(j,g*r)/h1n(j,g*rad)*j*(-sin(j*t+phi)))
	    
    Fz = Fz_in(R,T) * (R <= rad) + Fz_out(R,T) * (R > rad)
    Fr = Fr_in(R,T) * (R <= rad) + Fr_out(R,T) * (R > rad)
    Ft = Ft_in(R,T) * (R <= rad) + Ft_out(R,T) * (R > rad)
    
    Fx = Fr*cos(T) - Ft*cos(pi/2.-T)
    Fy = Fr*sin(T) + Ft*sin(pi/2.-T)
        
    # Normalization
    Fz /= amax(abs(Fz))
    #Fx /= amax(abs(Fz))
    #Fy /= amax(abs(Fz))

    extent = array([amin(x),amax(x),amin(y),amax(y)]) / rad
    cir = Circle((0,0),radius=1,facecolor='none',edgecolor='k',linewidth=2)
    fs = (6,4)
    save=True
    PATH = 'C:\\Users\\Patrick\\Documents\\PhD\\Presentations\\'
    
    fig=figure(figsize=fs)
    ax=fig.add_subplot(111)
    im = ax.imshow(abs(Fz)**2,extent=extent,interpolation='nearest')
    ax.add_patch(cir)
    fig.colorbar(im)
    ax.set_title(r'$|H_{z}|^2$')
    ax.set_xlabel('x/radius')
    ax.set_ylabel('y/radius')
    if save: savefig(PATH + 'TE' + str(j) + str(m)+'_Hz',dpi=80)

    fig=figure(figsize=fs)
    ax=fig.add_subplot(111)
    im = ax.imshow(Fx.imag,extent=extent,interpolation='nearest',cmap='RdBu')
    ax.add_patch(cir)
    fig.colorbar(im)
    ax.set_title(r'$E_{x}$')
    ax.set_xlabel('x/radius')
    ax.set_ylabel('y/radius')
    if save: savefig(PATH + 'TE' + str(j) + str(m)+'_Ex',dpi=80)

    fig=figure(figsize=fs)
    ax=fig.add_subplot(111)
    im = ax.imshow(Fy.imag,extent=extent,interpolation='nearest',cmap='RdBu')
    ax.add_patch(cir)
    fig.colorbar(im)
    ax.set_title(r'$E_{y}$')
    ax.set_xlabel('x/radius')
    ax.set_ylabel('y/radius')
    if save: savefig(PATH + 'TE' + str(j) + str(m)+'_Ey',dpi=80)

    fig=figure(figsize=fs)
    ax=fig.add_subplot(111)
    im = ax.imshow(abs(Fx)**2+abs(Fy)**2,extent=extent,interpolation='nearest')
    ax.add_patch(cir)
    fig.colorbar(im)

    interval = 10
    ax.quiver(X[::interval,::interval]/rad,Y[::interval,::interval]/rad,Fx.imag[::interval,::interval],Fy.imag[::interval,::interval],
				pivot='mid',color='k')
    ax.set_xlim(extent[0],extent[1])
    ax.set_ylim(extent[2],extent[3])
    ax.set_title(r'$|E|^2$')
    ax.set_xlabel('x/radius')
    ax.set_ylabel('y/radius')
    if save: savefig(PATH + 'TE' + str(j) + str(m)+'_Et',dpi=80)

    #show()


if __name__ == '__main__':
    
    
    a = linspace(25,250,500)*1e-9						#NW radius
    wl = linspace(400,800,10) * 1e-9					#Free Space Wavelength (nm)

    n = ones(len(wl))*3.4
    n = real(n) - 1j*imag(n)
    
    ResonanceMap(wl,a,n,pol='TE')
    #ResonanceMap(wl,a,n,no=2,pol='TE')
    
    
    #example_effectiveIndex()
    #ModePlot(3,0,4,phi=pi/2.)
    
    plt.show()
    
