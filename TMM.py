import numpy as np
import pylab as pl
import scipy as sp

## Constants ##
q = 1.6e-19
h = 6.626e-34
c = 3e8

## Mathematical Function Redefinitions ##
pi = np.pi
cos = np.cos
sin = np.sin
arcsin = np.arcsin
exp = np.exp
csqrt = np.lib.scimath.sqrt

## Magnitude Squared ##
def magsq(x): return (x * np.conj(x)).real

## Snells Law, Fresnel Equations and Phase Pickup ##
def snell(n1, n2, th): return arcsin(complex(n1 * sin(th) / n2))

def ts(ni, nf, thi, thf): return ((2 * ni * cos(thi)) / (ni * cos(thi) + nf * cos(thf)))
def tp(ni, nf, thi, thf): return ((2 * ni * cos(thi)) / (nf * cos(thi) + ni * cos(thf)))
def t(pol, ni, nf, thi, thf):
	if pol == 's': return ts(ni, nf, thi, thf)
	elif pol == 'p': return tp(ni, nf, thi, thf)

def rs(ni, nf, thi, thf): return ((ni * cos(thi) - nf * cos(thf)) / (ni * cos(thi) + nf * cos(thf)))
def rp(ni, nf, thi, thf): return ((nf * cos(thi) - ni * cos(thf)) / (nf * cos(thi) + ni * cos(thf)))
def r(pol, ni, nf, thi, thf):
	if pol == 's': return rs(ni, nf, thi, thf)
	elif pol == 'p': return rp(ni, nf, thi, thf)

def phase(n, th, d, wl): return (2 * pi * d * n * cos(th) / wl)

## 'M' Matrix Definition ##
#def getM(nm, n, th, d, wl, pol):
#	coeff = 1. / t(pol, nm, n, th, snell(nm, n, th))
#	A = exp(-1j * phase(n, th, d, wl))
#	B = r(pol, nm, n, th, snell(nm, n, th)) * exp(1j * phase(n, th, d, wl))
#	C = r(pol, nm, n, th, snell(nm, n, th)) * exp(-1j * phase(n, th, d, wl))
#	D = exp(1j * phase(n, th, d, wl))
#	M = np.matrix([[A,B],[C,D]])*coeff
#	return M
def getM(ni, nm, n, th, d, wl, pol):
	thi = snell(ni, nm, th)
	tho = snell(ni, n, th)
	coeff = 1. / t(pol, nm, n, thi, tho)
	A = exp(-1j * phase(n, tho, d, wl))
	B = r(pol, nm, n, thi, tho) * exp(1j * phase(n, tho, d, wl))
	C = r(pol, nm, n, thi, tho) * exp(-1j * phase(n, tho, d, wl))
	D = exp(1j * phase(n, tho, d, wl))
	M = np.matrix([[A,B],[C,D]])*coeff
	return M

#def getMlast(nm, n, th, pol):
#	coeff = 1. / t(pol, nm, n, th, snell(nm, n, th))
#	A = 1
#	B = r(pol, nm, n, th, snell(nm, n, th))
#	C = r(pol, nm, n, th, snell(nm, n, th))
#	D = 1
#	M = np.matrix([[A,B],[C,D]])*coeff
#	return M
def getMlast(ni, nm, n, th, pol):
	thi = snell(ni, nm, th)
	tho = snell(ni, n, th)
	coeff = 1. / t(pol, nm, n, thi, tho)
	A = 1
	B = r(pol, nm, n, thi, tho)
	C = r(pol, nm, n, thi, tho)
	D = 1
	M = np.matrix([[A,B],[C,D]])*coeff
	return M

## Reflectance and Transmittance ##
def R(M): return magsq(M[1,0]/M[0,0])
def T(M, n0, nt, th, pol): 
	if pol == 's': return magsq(1./M[0,0]) * (complex(nt).real * cos(snell(n0, nt, th))) / (n0 * cos(th))
	if pol == 'p': return magsq(1./M[0,0]) * (complex(nt).real * cos(snell(n0, nt, th))) / (n0 * cos(th))

#############################
### Solve Arbitrary Stack ###
#############################
def solvestack(ni, nt, ns, ds, wl, pol, th):
	"""
	Takes in array of refractive indices and corresponding thicknesses
	Returns R, T and A for entire stack
	Input:
	ni - Incident material refractive index
	nt - Transmitted material refractive index
	ns - Array of indices for remaining stack layers
	ds - Array of thicknesses corresponding to ns
	wl - Free space wavelength (in same units as ds)
	pol - Incident radiation polarization, either pol='s' or pol='p'
	th - Incident angle with respect to interface normal (th=0 is normal incidence)
	Notes:
	-ni and nt must represent semi-infinite layers
	-ni must be real valued, all other indices may be complex
	"""
	if pol == 's' or pol == 'p':
		M = getM(ni, ni, ns[0], th, ds[0], wl, pol)
		for iin in range(1, len(ns)):
			M = M * getM(ni, ns[iin-1], ns[iin], th, ds[iin], wl, pol)
		M = M * getMlast(ni, ns[-1], nt, th, pol)
		Ref = R(M)
		Tra = T(M, ni, nt, th, pol)
		Abso = 1. - (Ref + Tra)
		
		return Ref, Tra, Abso
	elif pol == 'u':
		pol = 's'
		M = getM(ni, ni, ns[0], th, ds[0], wl, pol)
		for iin in range(1, len(ns)):
			M = M * getM(ni, ns[iin-1], ns[iin], th, ds[iin], wl, pol)
		M = M * getMlast(ni, ns[-1], nt, th, pol)
		Ref_s = R(M)
		Tra_s = T(M, ni, nt, th, pol)
		Abs_s = 1. - (Ref_s + Tra_s)
		pol = 'p'
		M = getM(ni, ni, ns[0], th, ds[0], wl, pol)
		for iin in range(1, len(ns)):
			M = M * getM(ni, ns[iin-1], ns[iin], th, ds[iin], wl, pol)
		M = M * getMlast(ni, ns[-1], nt, th, pol)
		Ref_p = R(M)
		Tra_p = T(M, ni, nt, th, pol)
		Abs_p = 1. - (Ref_p + Tra_p)
		
		return 0.5*(Ref_s+Ref_p), 0.5*(Tra_s+Tra_p), 0.5*(Abs_s+Abs_p)
	else: print 'Invalid polarization!'

#######################
## Example Functions ##
#######################
def example_fp():
	ni = 1.0
	nfp = 20.0
	ns = pl.array([nfp])
	nt = 1.0
	
	wl = 500
	th = 0
	pol = 's'
	
	ds = pl.arange(0, 4*wl/nfp+1, wl/nfp/400)
	
	R = pl.zeros(len(ds), dtype='float')
	T = pl.zeros(len(ds), dtype='float')
	A = pl.zeros(len(ds), dtype='float')
	for id, d in enumerate(ds):
		R[id], T[id], A[id] = solvestack(ni, nt, ns, pl.array([d]), wl, pol, th)
	
	pl.figure(figsize=(10,7.5))
	pl.subplot(211)
	pl.plot(ds, T, 'b', lw=2, label='T')
	pl.xticks(ds[::100], ds[::100]/(wl/nfp))
	pl.xlim(ds[0], ds[-1])
	pl.ylim(0, 1)
	pl.title(r'Fabry-Perot Etalon')
	pl.ylabel(r'$T$', fontsize=18)
	pl.subplot(212)
	pl.plot(ds, R, 'r', lw=2, label='R')
	pl.xticks(ds[::100], ds[::100]/(wl/nfp))
	pl.xlim(ds[0], ds[-1])
	pl.ylim(0, 1)
	pl.ylabel(r'$R$', fontsize=18)
	pl.xlabel(r'$t_{slab} / \lambda$', fontsize=18)
	pl.show()
	
def example_mathematica1():
	## Units ##
	nm = 1e-9
	cm = 1e-2
	
	## Stack Definition ##
	ni = 1.+0j
	ns = pl.array([2.2, 3.3+0.3j])
	ds = pl.array([100*nm, 300*nm])
	nt = 1.+0j
	
	## Light Parameters ##
	ks = pl.linspace(1, 100000, 1001)
	wls = cm / ks
	
	Rs = np.zeros(len(wls), dtype='float')
	Ts = np.zeros(len(wls), dtype='float')
	As = np.zeros(len(wls), dtype='float')
	
	## Normal Incidence ##
	th = 0
	for iwl, wl in enumerate(wls):
		Rs[iwl], Ts[iwl], As[iwl] = solvestack(ni, nt, ns, ds, wl, 'u', th)
	Ru_Normal = Rs
	
	Rs = np.zeros(len(wls), dtype='float')
	Ts = np.zeros(len(wls), dtype='float')
	As = np.zeros(len(wls), dtype='float')
	
	## pi/4 Incidence ##
	th = np.pi / 4
	for iwl, wl in enumerate(wls):
		Rs[iwl], Ts[iwl], As[iwl] = solvestack(ni, nt, ns, ds, wl, 'u', th)
	Ru_PIov4 = Rs
	
	## Plot Results ##
	pl.figure(figsize=(10,7.5))
	pl.plot(ks, Ru_Normal, 'b', lw=2)
	pl.plot(ks, Ru_PIov4, 'r', lw=2)
	pl.ylabel('Fraction Reflected')
	pl.xlabel('Wavenumber (cm^-1)')
	pl.title('Reflection of unpolarized light at normal incidence (blue), pi/4 incidence (red)')	
	pl.show()

def example_spp():
	nm = 1e-9
	um = 1e-6
	cm = 1e-3
	
	ni = csqrt(12.295)
	ns = np.array([csqrt(2.66)])
	nt = csqrt(-70.72+7.06J)
	
	wl = 1.319 # um
	ths = np.linspace(0.0, 0.5*pi, 1000)
	pol = 'p'
	
	ds = np.array([1.2]) # um
	
	R = pl.zeros(len(ths), dtype='float')
	T = pl.zeros(len(ths), dtype='float')
	A = pl.zeros(len(ths), dtype='float')
	for ith, th in enumerate(ths):
		R[ith], T[ith], A[ith] = solvestack(ni, nt, ns, ds, wl, pol, th)
		
	pl.figure(figsize=(10,7.5))
	pl.plot(ths, R, 'r', lw=2, label='R')
	#pl.xlim(ths[0], ths[-1])
	#pl.ylim(0, 1)
	pl.ylabel(r'$R$', fontsize=18)
	pl.xlabel(r'$\theta$', fontsize=18)
	pl.show()
	
