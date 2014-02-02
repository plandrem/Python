#!/usr/bin/python
from __future__ import division
import pylab, jfdfd, jfdfdUtil, numpy, sys, os, pickle
import pp
import materials
import Plasmons
import plots
from jfdfd import TEz as TE, TMz as TM
import jFDFD_to_NTFF
import pickle
import Patrick_Utilities as putil

parallel = True

def getScatteredField(d,subkeys):
	
	'''
	Loads jFDFD grid objects and manipulates them to return the scattered field from a simulation.
	'inc' refers to a simulation with only an excitation source in free space.  
	'sub' inlcudes some background structure such as a substrate, but not the scatterer of interest.
	'tot' includes the total problem.
	
	if TFSF sourcing was used (which implies no inc or sub files exist) the incident field is 
	calculated by transfer matrix and subtracted.
	'''
	# Load fields	
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)	
	inc_name = 'fields/inc-' + fnbase + '.h5'
	sub_name = 'fields/sub-' + fnbase + '.h5'
	tot_name = 'fields/tot-' + fnbase + '.h5'
	
	if os.path.exists(os.path.join(os.getcwd(), tot_name)):
		
		if d.inc: g_inc = jfdfd.h5inputGrid2D(inc_name)
		if d.sub: g_sub = jfdfd.h5inputGrid2D(sub_name)
		g_tot = jfdfd.h5inputGrid2D(tot_name)
		
		# Set coordinates for flux calculation
		Lx, Ly = jfdfdUtil.get_grid_size(g_tot)
		nx = g_tot.nx
		ny = g_tot.ny
		
		# Subtract incident field from total
		if d.sub:
			for i in range(nx):
				for j in range(ny):
					Fz_sub = jfdfd.getFz(g_sub,i,j)
					Fz_tot = jfdfd.getFz(g_tot,i,j)
					jfdfd.setFz(g_tot,i,j,Fz_tot - Fz_sub)
					
			jfdfd.computeComplimentaryFields(g_tot)
		
		elif d.inc:
			for i in range(nx):
				for j in range(ny):
					Fz_inc = jfdfd.getFz(g_inc,i,j)
					Fz_tot = jfdfd.getFz(g_tot,i,j)
					jfdfd.setFz(g_tot,i,j,Fz_tot - Fz_inc)
					
			jfdfd.computeComplimentaryFields(g_tot)
			
		#elif d.source == 'TFSF':
		'''
		ideally the total field region would be completely subtracted here.
		'''
		
		output = [g_tot]
		if d.inc: output.append(g_inc)
		if d.sub: output.append(g_sub)
		return output
		
	else:
		print 'Simulation files do not exist!'
		return 0

def calc_max_field_one(d, subkeys):
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/max - ' + fnbase + '.dat'
	
	#if os.path.exists(os.path.join(os.getcwd(), sname)):
		#absval = numpy.loadtxt('data/raw/max - ' + fnbase + '.dat')
		#if absval[0]!=0 and absval[1]!=0:
			#print 'Analyzed data already exists - ', fnbase
			#return
			
	tot_name = 'fields/tot-' + fnbase + '.h5'
	print fnbase
	if os.path.exists(os.path.join(os.getcwd(), tot_name)):
		
		d = putil.LoadFields(tot_name)
		max = pylab.amax(abs(d['Fz']))
		
	else:
		print 'Simulation files do not exist!'
		max = 0
	
	# Save data
	xsections = pylab.zeros(1, 'd')
	xsections[0] = max
	
	print "Max Field - ", max
	
	numpy.savetxt(sname, xsections)
	
	return max
	



def calc_max_field(rparams, subkeys):
	"""
	Submits jobs to job server
	"""
	results = []
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_max_field_one, (rd, subkeys), (), ('jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials','Patrick_Utilities as putil'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			results.append(job())
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			results.append(calc_max_field_one(rd, subkeys))
			
	pylab.plot(results)
	pylab.show()

def calc_abs_one(d, subkeys,redo=False):
	"""
	Calculates absorption all layers
	"""
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/abs - ' + fnbase + '.dat'
	
	if os.path.exists(os.path.join(os.getcwd(), sname)):
		absval = numpy.loadtxt('data/raw/abs - ' + fnbase + '.dat')
		if not redo and absval[0]!=0 and absval[1]!=0:
			print 'Analyzed data already exists - ', fnbase
			return
			
	inc_name = 'fields/inc-' + fnbase + '.h5'
	tot_name = 'fields/tot-' + fnbase + '.h5'
	print fnbase
	#if os.path.exists(os.path.join(os.getcwd(), inc_name)) and os.path.exists(os.path.join(os.getcwd(), tot_name)):
	if os.path.exists(os.path.join(os.getcwd(), tot_name)):
		
		if d.inc: g_inc = jfdfd.h5inputGrid2D(inc_name)
		g_tot = jfdfd.h5inputGrid2D(tot_name)

		fd = putil.LoadFields(tot_name,PMLpad=-19)	
		
		
		# Set coordinates for flux calculation
		Lx, Ly = jfdfdUtil.get_grid_size(g_tot)
		nx = g_tot.nx
		ny = g_tot.ny
		
		resonator = putil.LoadResonator(tot_name)
		
		res_ymin = resonator['res_ymin']
		res_ymax = resonator['res_ymax']
		res_xmin = resonator['res_xmin']
		res_xmax = resonator['res_xmax']
		
		theta = 45.
		cross_section = (res_xmax - res_xmin)*sp.cos(theta*sp.pi/180.) + (res_ymax - res_ymin)*sp.sin(theta*sp.pi/180.)
		
		# Calculate incident flux per unit length
		if d.inc:
			x0f = res_xmin
			x1f = res_xmax
			y0f = int(Ly/2.)
			f1 = jfdfd.calculateFluxHorizontalC(g_inc, x0f, x1f, y0f)
			
			flux_inc = -f1 / (cross_section)
		else:
			flux_inc = 0.5
		
		# Calculate absorption cross-section in resonator
		x0a = res_xmin
		x1a = res_xmax
		y0a = res_ymin
		y1a = res_ymax

		abs_res = jfdfd.calculateOhmicAbsorptionRectangleC(g_tot, x0a, y0a, x1a, y1a) / (flux_inc*cross_section)

		if d.inc: jfdfd.freeGrid2D(g_inc)
		jfdfd.freeGrid2D(g_tot)
		
	else:
		print 'Simulation files do not exist!'
		flux_inc = 0
		abs_res = 0	
	
	# Save data
	xsections = pylab.zeros(2, 'd')
	xsections[0] = flux_inc
	xsections[1] = abs_res
	
	print "Incident flux - ", flux_inc
	print "Res Abs - ", abs_res
	
	numpy.savetxt(sname, xsections)
	



def calc_abs(rparams, subkeys,redo=False):
	"""
	Submits jobs to job server
	"""
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_abs_one, (rd, subkeys, redo), (), ('jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials','Patrick_Utilities as putil'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			job()
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			calc_abs_one(rd, subkeys, redo)

def calc_trans_one(d, subkeys,redo=False):
	"""
	Calculates transmission just above bottom PML layer
	"""
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/trans - ' + fnbase + '.dat'
	
	if os.path.exists(os.path.join(os.getcwd(), sname)):
		transval = numpy.loadtxt('data/raw/trans - ' + fnbase + '.dat')
		if not redo and transval[0]!=0 and transval[1]!=0:
			print 'Analyzed data already exists - ', fnbase
			return
			
	inc_name = 'fields/inc-' + str(d.wavelength) + '.h5'
	tot_name = 'fields/tot-' + fnbase + '.h5'
	print fnbase

	if os.path.exists(os.path.join(os.getcwd(), tot_name)):
		
		g_inc = jfdfd.h5inputGrid2D(inc_name)
		g_tot = jfdfd.h5inputGrid2D(tot_name)		
		
		# Set coordinates for flux calculation
		Lx, Ly = jfdfdUtil.get_grid_size(g_tot)
		nx = g_tot.nx
		ny = g_tot.ny
		
		# Calculate incident flux per unit length
		nxi = g_inc.nx
		nyi = g_inc.ny

		i0f = 0
		i1f = nxi
		j0f = int(nyi/2.)
		f1 = jfdfd.calculateFluxHorizontal2(g_inc, i0f, i1f, j0f)
		
		Lx_inc, Ly_inc = jfdfdUtil.get_grid_size(g_inc)
		flux_inc = -f1/Lx_inc
		
		# Calculate transmission just above PML
		i0 = 0
		i1 = nx
		j0 = d.PML + 5

		trans = -1 * jfdfd.calculateFluxHorizontal2(g_tot, i0, i1, j0) / Lx / flux_inc
		
		jfdfd.freeGrid2D(g_inc)
		jfdfd.freeGrid2D(g_tot)
		
	else:
		print 'Simulation files do not exist!'
		flux_inc = 0
		trans = 0	
	
	# Save data
	xsections = pylab.zeros(2, 'd')
	xsections[0] = flux_inc
	xsections[1] = trans
	
	print "Transmission - ", trans
	
	numpy.savetxt(sname, xsections)
	



def calc_trans(rparams, subkeys,redo=False):
	"""
	Submits jobs to job server
	"""
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_trans_one, (rd, subkeys, redo), (), ('jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials','Patrick_Utilities as putil'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			job()
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			calc_trans_one(rd, subkeys, redo)

def calc_ref_one(d, subkeys,redo=False):
	"""
	Calculates reflection just above bottom PML layer
	"""
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/ref - ' + fnbase + '.dat'
	
	if os.path.exists(os.path.join(os.getcwd(), sname)):
		refval = numpy.loadtxt('data/raw/ref - ' + fnbase + '.dat')
		if not redo and refval[0]!=0 and refval[1]!=0:
			print 'Analyzed data already exists - ', fnbase
			return
			
	inc_name = 'fields/inc-' + str(d.wavelength) + '.h5'
	sub_name = 'fields/sub-' + fnbase + '.h5'
	tot_name = 'fields/tot-' + fnbase + '.h5'
	print fnbase

	if os.path.exists(os.path.join(os.getcwd(), tot_name)):
		
		if d.inc: g_inc = jfdfd.h5inputGrid2D(inc_name)
		if d.sub: g_sub = jfdfd.h5inputGrid2D(sub_name)
		g_tot = jfdfd.h5inputGrid2D(tot_name)		
		
		# Set coordinates for flux calculation
		Lx, Ly = jfdfdUtil.get_grid_size(g_tot)
		nx = g_tot.nx
		ny = g_tot.ny
		
		# subtract incident plane wave
		if d.inc:
			for i in range(nx):
				for j in range(ny):
					Fz_inc = jfdfd.getFz(g_inc,i,j)
					Fz_tot = jfdfd.getFz(g_tot,i,j)
					jfdfd.setFz(g_tot,i,j,Fz_tot - Fz_inc)
					
			jfdfd.computeComplimentaryFields(g_tot)

		# Calculate incident flux per unit length
		if d.inc:
			nxi = g_inc.nx
			nyi = g_inc.ny

			i0f = 0
			i1f = nxi
			j0f = int(nyi/2.)
			f1 = jfdfd.calculateFluxHorizontal(g_inc, i0f, i1f, j0f)
			
			Lx_inc, Ly_inc = jfdfdUtil.get_grid_size(g_inc)
			flux_inc = -f1/Lx_inc
		else:
			flux_inc = -0.5
			
		# Calculate reflection just above PML
		i0 = 0
		i1 = nx
		j0 = ny - d.PML - 15

		ref = jfdfd.calculateFluxHorizontal(g_tot, i0, i1, j0) / Lx / flux_inc
		
		if d.inc: jfdfd.freeGrid2D(g_inc)
		if d.sub: jfdfd.freeGrid2D(g_sub)
		jfdfd.freeGrid2D(g_tot)
		
	else:
		print 'Simulation files do not exist!'
		flux_inc = 0
		ref = 0	
	
	# Save data
	xsections = pylab.zeros(2, 'd')
	xsections[0] = flux_inc
	xsections[1] = ref
	
	print "reflection - ", ref
	
	numpy.savetxt(sname, xsections)
	



def calc_ref(rparams, subkeys,redo=False):
	"""
	Submits jobs to job server
	"""
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_ref_one, (rd, subkeys, redo), (), ('jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials','Patrick_Utilities as putil'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			job()
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			calc_ref_one(rd, subkeys, redo)


def calc_sca_one(d, subkeys,redo=False,tight=False):
	"""
	Calculates Qsca by collecting all flux leaving boundaries of the simulation.  Works for dipole
	simulations where the incident plane wave is subtracted, or for TFSF simulations.
	
	Options:
	tight - flux box placed immediately outside of resonator.  Useful for problems with lossy substrates.
	"""
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/sca - ' + fnbase + '.dat'
	
	if os.path.exists(os.path.join(os.getcwd(), sname)):
		scaval = numpy.loadtxt('data/raw/sca - ' + fnbase + '.dat')
		if not redo and scaval[0]!=0 and scaval[1]!=0:
			print 'Analyzed data already exists - ', fnbase
			return
			
	inc_name = 'fields/inc-' + fnbase + '.h5'
	sub_name = 'fields/sub-' + fnbase + '.h5'
	tot_name = 'fields/tot-' + fnbase + '.h5'
	print fnbase
	
	if os.path.exists(os.path.join(os.getcwd(), tot_name)):
		
		if d.inc: g_inc = jfdfd.h5inputGrid2D(inc_name)
		if d.sub: g_sub = jfdfd.h5inputGrid2D(sub_name)
		g_tot = jfdfd.h5inputGrid2D(tot_name)
		
		#fd = putil.LoadFields(tot_name,PMLpad=15)	
		
		
		# Set coordinates for flux calculation
		Lx, Ly = jfdfdUtil.get_grid_size(g_tot)
		nx = g_tot.nx
		ny = g_tot.ny
		
		#res_steps = d.res_steps
		#PML = d.PML
		
		resonator = putil.LoadResonator(tot_name)
		
		res_imin = resonator['res_imin']
		res_imax = resonator['res_imax']
		res_jmin = resonator['res_jmin']
		res_jmax = resonator['res_jmax']
		res_xmin = resonator['res_xmin']
		res_xmax = resonator['res_xmax']
		
		cross_section = res_xmax - res_xmin
		
		# Calculate incident flux per unit length
		if d.inc:
			i0f = res_imin
			i1f = res_imax
			j0f = int(ny/2.)
			f1 = jfdfd.calculateFluxHorizontal(g_inc, i0f, i1f, j0f)
			
			flux_inc = -f1 / cross_section
		else:
			flux_inc = 0.5											# power density S = |E|^2/2Z, Z=1, |E|=1
			pass
			
		# Subtract incident field from total
		if d.sub:
			for i in range(nx):
				for j in range(ny):
					Fz_sub = jfdfd.getFz(g_sub,i,j)
					Fz_tot = jfdfd.getFz(g_tot,i,j)
					jfdfd.setFz(g_tot,i,j,Fz_tot - Fz_sub)
					
			jfdfd.computeComplimentaryFields(g_tot)
		
		elif d.inc:
			for i in range(nx):
				for j in range(ny):
					Fz_inc = jfdfd.getFz(g_inc,i,j)
					Fz_tot = jfdfd.getFz(g_tot,i,j)
					jfdfd.setFz(g_tot,i,j,Fz_tot - Fz_inc)
					
			jfdfd.computeComplimentaryFields(g_tot)
			
		# Calculate scattering cross-section in resonator
		if not tight:
			pad = 10
			i0a = d.PML+pad
			i1a = nx-d.PML-pad
			j0a = d.PML+pad
			j1a = ny-d.PML-pad

		else:
			pad = 2
			i0a = res_imin - pad
			i1a = res_imax + pad
			j0a = res_jmin-pad											# no pad -- don't want box to penetrate substrate if lossy
			j1a = res_jmax + pad

		up = jfdfd.calculateFluxHorizontal(g_tot,i0a,i1a,j1a)
		down = jfdfd.calculateFluxHorizontal(g_tot,i0a,i1a,j0a)
		left = jfdfd.calculateFluxVertical(g_tot,j0a,j1a,i0a)
		right = jfdfd.calculateFluxVertical(g_tot,j0a,j1a,i1a)
		
		print up, down, left, right
		Psca = up-down-left+right
		
		
		Qsca = Psca / (flux_inc*cross_section)
		
		if d.inc: jfdfd.freeGrid2D(g_inc)
		if d.sub: jfdfd.freeGrid2D(g_sub)
		jfdfd.freeGrid2D(g_tot)

		#PlotBox(tot_name,i0a,i1a,j0a,j1a)
		#exit()
		
	else:
		print 'Simulation files do not exist!'
		flux_inc = 0
		Qsca = 0	
	
	# Save data
	xsections = pylab.zeros(2, 'd')
	xsections[0] = flux_inc
	xsections[1] = Qsca
	
	print "Incident flux -", flux_inc
	print "Psca -", Psca
	print "Qsca -", Qsca
	
	numpy.savetxt(sname, xsections)
	



def calc_sca(rparams, subkeys,redo=False):
	"""
	Submits jobs to job server
	"""
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_sca_one, (rd, subkeys, redo), (), ('jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials','Patrick_Utilities as putil'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			job()
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			calc_sca_one(rd, subkeys, redo)

def calc_power_conversion_one(d, subkeys, redo=False):
	"""
	Calculates SPP power conversion efficiency
	"""
	
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/pce - ' + fnbase + '.dat'
	
	if os.path.exists(os.path.join(os.getcwd(), sname)):
		absval = numpy.loadtxt('data/raw/pce - ' + fnbase + '.dat')
		if not redo and absval.all()!=0:
			print 'Analyzed data already exists -', fnbase
			return
				
	tot_name = 'fields/tot-' + fnbase + '.h5'
	print fnbase
	
	if d.source == 'dipole':
		gs = getScatteredField(d,subkeys)
		g_tot = gs[0]
		
	else: g_tot = jfdfd.h5inputGrid2D(tot_name)
	
	resonator = putil.LoadResonator(tot_name)
	
	res_imin = resonator['nw_imin']
	res_imax = resonator['nw_imax']
	res_jmin = resonator['nw_jmin']
	res_jmax = resonator['nw_jmax']
	res_xmin = resonator['nw_xmin']
	res_xmax = resonator['nw_xmax']
	
	if d.inc:
		g_inc = gs[1]
		i0f = res_imin
		i1f = res_imax
		j0f = int(g_tot.ny/2.)
		f1 = jfdfd.calculateFluxHorizontal(g_inc, i0f, i1f, j0f)
		
		flux_inc = -f1 / (res_xmax-res_xmin)
	else:
		flux_inc = 0.5
		
	if d.sub: g_sub = gs[2]

	if os.path.exists(os.path.join(os.getcwd(), tot_name)):
		#try:
		PCE = Plasmons.GetPlasmons_TFSF(tot_name,flux_inc=flux_inc)
		#except:
		#	os.remove(tot_name)
		
	else:
		print 'Simulation files do not exist!'
		PCE = 0
	
	# Save data
	xsections = pylab.zeros(1, 'd')
	xsections[0] = PCE
	
	numpy.savetxt(sname, xsections)
	
	if d.inc: jfdfd.freeGrid2D(g_inc)
	if d.sub: jfdfd.freeGrid2D(g_sub)
	jfdfd.freeGrid2D(g_tot)



def calc_power_conversion(rparams, subkeys, redo=False):
	"""
	Submits jobs to job server
	"""
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_power_conversion_one, (rd, subkeys, redo), (getScatteredField,), ('Patrick_Utilities as putil','jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials', 'Plasmons'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			job()
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			calc_power_conversion_one(rd, subkeys, redo)

def calc_pce_thin_one(d, subkeys):
	"""
	Calculates SPP power conversion efficiency
	"""
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/pce - ' + fnbase + '.dat'
	
	if os.path.exists(os.path.join(os.getcwd(), sname)):
		absval = numpy.loadtxt('data/raw/pce - ' + fnbase + '.dat')
		if absval.all()!=0:
			print 'Analyzed data already exists -', fnbase
			return

	fname = 'fields/' + fnbase + '.h5'
	if os.path.exists(os.path.join(os.getcwd(), fname)):
		PCE = Plasmons.Get_Leakage(fname)
		
	else:
		print 'Simulation files do not exist!'
		PCE = 0
	
	# Save data
	xsections = pylab.zeros(1, 'd')
	xsections[0] = PCE
	
	numpy.savetxt(sname, xsections)


def calc_pce_thin(rparams, subkeys):
	"""
	Submits jobs to job server
	"""
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_pce_thin_one, (rd, subkeys), (), ('jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials', 'Plasmons'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			job()
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			calc_pce_thin_one(rd, subkeys)
			
def calc_FF_one(d, subkeys):
	"""
	Calculates scattered FF power normalized such that max scattered power in each hemisphere = 1
	"""
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/ff - ' + fnbase + '.dat'
	
	if os.path.exists(os.path.join(os.getcwd(), sname)):
		absval = numpy.loadtxt('data/raw/ff - ' + fnbase + '.dat')
		if absval.all()!=0:
			print 'Analyzed data already exists -', fnbase
			return

	fname = 'fields/tot-' + fnbase + '.h5'
	if os.path.exists(os.path.join(os.getcwd(), fname)):
		
		interp = 80
		ntheta = 201													# number of polar angles to calculate
		atheta = pylab.linspace(-88,88,ntheta)/180*pylab.pi							# range of polar angles
		
		top, bottom = jFDFD_to_NTFF.NTFF(fname,atheta=atheta,interp=interp,PMLpad=15)
		
		norm = 1 #pylab.sum(top) + pylab.sum(bottom)
		top = top/norm
		bottom = bottom/norm
		
	else:
		print 'Simulation files do not exist!'
		top = bottom = 0
	
	# Save data
	xsections = pylab.zeros((2,len(top)), 'd')
	xsections[0] = top
	xsections[1] = bottom
	
	numpy.savetxt(sname, xsections)


def calc_FF(rparams, subkeys):
	"""
	Submits jobs to job server
	"""
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_FF_one, (rd, subkeys), (), ('jFDFD_to_NTFF','jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials', 'Plasmons'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			job()
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			calc_FF_one(rd, subkeys)

def calc_pce_fft_one(d, subkeys):
	"""
	Calculates SPP power conversion efficiency
	"""
	
	# Load fields
	fnbase = jfdfdUtil.generate_filename(d, subkeys=subkeys)
	sname = 'data/raw/pce - ' + fnbase + '.dat'
	
	if os.path.exists(os.path.join(os.getcwd(), sname)):
		absval = numpy.loadtxt('data/raw/pce - ' + fnbase + '.dat')
		if absval.all()!=0:
			print 'Analyzed data already exists -', fnbase
			return

	fname = 'fields/tot-' + fnbase + '.h5'
	if os.path.exists(os.path.join(os.getcwd(), fname)):
		PCE = Plasmons.GetLeakage_FFT(fname,pad=1e4,res=20)
		
	else:
		print 'Simulation files do not exist!'
		PCE = 0
	
	# Save data
	xsections = pylab.zeros(1, 'd')
	xsections[0] = PCE
	
	numpy.savetxt(sname, xsections)


def calc_pce_fft(rparams, subkeys):
	"""
	Submits jobs to job server
	"""
	
	# get a list of all dictionaries we want to run...
	rdicts = jfdfdUtil.dictpairs(rparams)
	
	if parallel:
		# tell parallel python how many cpus to use
		job_server = pp.Server(ncpus=rparams['ncpus'])
		jobs = []
		
		# go through the dictionaries and run them or submit them to the job server..
		for rd in rdicts:
			#run_one(rd, subkeys)
			job = job_server.submit(calc_pce_fft_one, (rd, subkeys), (), ('jfdfd', 'jfdfdUtil', 'pylab', 'numpy', 'materials', 'Plasmons','Patrick_Utilities','plasmon_fft'))
			jobs.append(job)
		njobs = len(jobs)
		print 'Submitted %d jobs' % njobs
		
		for ijob, job in enumerate(jobs):
			job()
			print ' %g%% done (%d of %d jobs remaining)' % ((ijob+1)*100./njobs, njobs-ijob-1, njobs)
	else:
		for rd in rdicts:
			calc_pce_fft_one(rd, subkeys)
			
def PlotBox(tot_name,i0,i1,j0,j1):
	'''
	Plots primary (z) field in grid g with overlaid rectangle (corresponding to scattering/absorption box).  
	The rectangle corners are (i0,j0) to (i1,j1).
	'''
	
	fd = putil.LoadFields(tot_name)
	
	i0 -= 20
	i1 -= 20
	j0 -= 20
	j1 -= 20
	
	fig = pylab.figure()
	ax = fig.add_subplot(111)
	
	im = ax.imshow(fd['eps'].real)
	
	box = ax.add_patch(pylab.Rectangle((i0,j0),(i1-i0),(j1-j0), fill=False, edgecolor='white', linestyle='dashed',linewidth=3))
	pylab.show()
	
if __name__=='__main__':
	
	'''
	Generates plots from h5 files.
	
	List of Command Line Arguments:
	
	1) Mode (pce or abs): sets plasmon conversion efficiency or absorption plot
	
	2) Dependent variable
	
	3) 2nd dependent variable (produces 2d colormap) - must be entered or 'none' if using additional arguments
	
	Options:
	
	(-slice) 	If running a simulation over more than 2 degrees of freedom, this allows the extra variables to be fixed.  
				Format is key=value.  If 2nd dependent variable is 'none', value may be '?' to set 
				key to 2nd dependent variable.
				
	(-skip)		Skips data analysis and plots previously computed results
	
	'''
	
	if len(sys.argv) > 1: mode = str(sys.argv[1])
	else: mode = 'pce'
	
	if len(sys.argv) > 2: parameter = str(sys.argv[2])
	else: parameter = 'wavelength'
	
	if len(sys.argv) > 3: parameter2 = str(sys.argv[3])
	else: parameter2 = None
	
	if parameter2 == 'none': parameter2 = None
						
	# load rparams from file
	with open(os.getcwd() + '/rparams.csv','r') as f:
		rparams = pickle.load(f)

	#load subkeys from file
	with open(os.getcwd() + '/subkeys.csv','r') as f:
		subkeys = pickle.load(f)
		
	if '-slice' in sys.argv:
		i = sys.argv.index('-slice')
		for n, arg in enumerate(sys.argv[i+1:]):
			
			try:
				argname,argval = arg.split('=')
				
				print type(rparams[argname][0])
				if argval == '?': parameter2 = argname
				elif argval == 'c': rparams[argname] = 'c'
				elif argval == 'a': rparams[argname] = 'a'
				elif argname == 'angle': rparams[argname] = float(argval)
				elif type(rparams[argname][0])==numpy.float64 or type(rparams[argname][0])==float: rparams[argname] = float(argval)
				elif type(rparams[argname][0])==numpy.int64 or type(rparams[argname][0])==int: rparams[argname] = int(argval)
				else: rparams[argname] = float(argval)
				
			except:
				break
				
	if '-skip' in sys.argv: skip=True
	else: skip=False
				
	if '-av' in sys.argv: average=True
	else: average=False
				
	if '-norm' in sys.argv: norm=True
	else: norm=False
	
	if '-redo' in sys.argv: redo=True
	else: redo=False
				
	if '-line' in sys.argv or parameter2 == None: showLinePlot=True
	else: showLinePlot=False	
	
	if '-noplot' in sys.argv: noplot=True
	else: noplot=False

	if '-log' in sys.argv: logplot=True
	else: logplot=False

	if not skip:
		if mode == 'pce':
			calc_power_conversion(rparams = rparams, subkeys = subkeys, redo = redo)
			#calc_pce_thin(rparams = rparams, subkeys = subkeys)
			#calc_pce_fft(rparams = rparams, subkeys = subkeys)
		
		if mode == 'abs':
			calc_abs(rparams = rparams, subkeys = subkeys,redo=redo)
			
		if mode == 'sca':
			calc_sca(rparams = rparams, subkeys = subkeys,redo=redo)
			
		if mode == 'trans':
			calc_trans(rparams = rparams, subkeys = subkeys,redo=redo)
			
		if mode == 'ref':
			calc_ref(rparams = rparams, subkeys = subkeys,redo=redo)
			
		if mode == 'ext':
			calc_abs(rparams = rparams, subkeys = subkeys,redo=redo)
			calc_sca(rparams = rparams, subkeys = subkeys,redo=redo)
			showLinePlot = True
			
		if mode == 'sppext':
			calc_abs(rparams = rparams, subkeys = subkeys,redo=redo)
			calc_power_conversion(rparams = rparams, subkeys = subkeys, redo = redo)

		if mode == 'ff':
			calc_FF(rparams = rparams, subkeys = subkeys)
			plots.plotFF(rparams, subkeys, parameter=parameter, save=True)
		
		if mode == 'max':
			calc_max_field(rparams = rparams, subkeys = subkeys)

	#showLinePlot = True
	show2Dplot = not showLinePlot

	if not noplot:
		plots.plot(rparams,
			subkeys,
			mode,
			parameter=parameter,
			parameter2=parameter2,
			showLinePlot=showLinePlot,
			show2Dplot=show2Dplot,
			average=average,
			norm=norm,
			logplot=logplot,
			save=False
			)
			
