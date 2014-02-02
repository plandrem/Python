#!/usr/bin/python

from __future__ import division

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt

import putil

from fractions import Fraction

def getSigns(n):
	
	'''
	return a list of n random +'s and -'s
	'''
	
	nums = np.rint(np.random.random(n)).astype(int)
	signs = []
	
	for s in nums:
		if s == 0: signs.append('+')
		else: signs.append('-')
		
	return signs
	
def sign2num(s):
	
	'''
	converts string polarity (+,-) into multiplier (1,-1)
	'''
	nums = []
	
	if len(s)>1:
		for si in s:
			if si == '+': nums.append(1)
			else: nums.append(-1)
			
		return nums
			
	else:
		if s == '+': return 1
		else: return -1
		
def num2sign(n):
	'''
	returns polarity of n as string
	'''
	
	if n >= 0: return '+'
	elif n < 0: return '-'
	
def getFrac(neg = True):
    
    '''
    generates random fraction, reduces to simplest form.  Returns (num,denom).
    neg = False disables negative fractions
    '''
    a,b = np.rint(np.random.random(2) * 10 + 1).astype(int)
    sign = np.rint(np.random.random(1)).astype(int)
    
    if sign == 0 and neg: s = -1
    else: s = 1
    
    f = Fraction(a,b)
    
    return (s * f.numerator, f.denominator)
    
def getFracs(n=1,neg=True,zero=True):
	'''
	returns list of n fraction objects
	neg allows negative values
	zero allows zero to be returned.  If false, numerator replaced by 1
	'''
	
	Fs = []
	
	for i in range(n):
		a,b = np.random.randint(-10,10,2)
		if b == 0: b = 1
		if a == 0 and not zero: a = 1
		
		f = Fraction(a,b)
		
		if neg==False:
			f = abs(f)
		
		Fs.append(f)
		
	return Fs
    
def simpleFractions(op='+',integers=True):
    
    '''
    integers forces one of the two fractions involved to be a whole number
    '''
    
    b=d=1                   # eliminate totally integer problems
    while b==1 and d==1:
        a,b = getFrac()
        c,d = getFrac()
  
    if integers:
        if b!=1 and d!=1:
            r = np.rint(np.random.random(1))
            if r==0: b=1
            elif r==1: d=1
        
    if op == '+':
        a = abs(a)
        c = abs(c)
        F = Fraction(a*d + b*c,b*d)

    if op == '-':
        a = abs(a)
        c = abs(c)
        F = Fraction(a*d - b*c,b*d)

    if op == '*':
        F = Fraction(a*c,b*d)

    if op == '/':
        F = Fraction(a*d,b*c)

    if b != 1: f1 = '%u/%u' % (a,b)
    else: f1 = str(a)

    if d != 1: f2 = '%u/%u' % (c,d)
    else: f2 = str(c)
    
    prob = f1 + ' ' + op + ' ' + f2 + ' = ?'

    if F.denominator != 1: sol = '%u/%u' % (F.numerator,F.denominator)
    else: sol = str(F.numerator)
    
    return prob,sol
    
def test_simpleFractions():
    
    ops = ['+','-','*','/']
    
    for i in range(4):
        print ops[i]
        print
        for n in range(3):
            p,s = simpleFractions(op = ops[i])
            print p
            print s
            print
            
def LineGraph():
    
    '''
    presents a problem of form "Ay + Bx + C = Dx" with some scrambling of terms.
    Student should solve for y, find slope, y intercept, x intercept, and draw the graph.
    '''
    
    # Get coefficients
    A = getFrac()
    B = getFrac()
    C = getFrac()
    D = getFrac()
    
    coef = [A,B,C,D]
    var = ['y','x','x','']
    
    #pos stores the positions of elements in the equation - 0-3 are coefficients, 4 is '='    
    pos = np.arange(5)

    # make sure equals is not first element
    while pos[0] == 4 or pos[4] == 4: np.random.shuffle(pos)

    terms = []
    
    for i in range(5):
        
        p = pos[i]
        if p != 4:
            if coef[p][0] >= 0: sign = '+'
            elif coef[p][0] < 0: sign = '-'
            
            # additional case for future - eliminate 1's in numerator if variable present
            if coef[p][1]==1:
                if i==0:                    
                    term = str(coef[p][0]) + var[p]
                elif pos[i-1]==4:                    
                    term = ' ' + str(coef[p][0]) + var[p]
                else:
                    term = ' ' + sign + ' ' + str(abs(coef[p][0])) + var[p]
            else: 
                if i==0:
                    term = ' %u' % coef[p][0] + var[p] + '/%u' % coef[p][1]
                elif pos[i-1]==4:
                    term = ' %u' % coef[p][0] + var[p] + '/%u' % coef[p][1]
                else:
                    term = ' ' + sign + ' %u' % abs(coef[p][0]) + var[p] + '/%u' % coef[p][1]
        else:
            term = ' ='
               
        terms.append(term)
    
    prob = ''.join(terms)
    print prob
    
    #Solve

    # adjust signs to account for equals sign placement

    fracs = []
    
    for i,x in enumerate(coef):
        
        m,n = coef[i]
        
        if np.where(pos==4) < np.where(pos==i):
            m *= -1
            
        fracs.append(Fraction(m,n))
    
    A,B,C,D = fracs        
    
    m = -1*(B+C)/A
    b = -1*D/A
    
    if m != 0: xint = -b/m
    else: xint = 'None'
    
    soln = 'm: ' + str(m) + '\n' + 'b: ' + str(b) + '\n' + 'x int: ' + str(xint) + '\n'
    
    
    print soln
    
    return prob, soln

def test_LineGraph():

    for n in range(3):
        p,s = LineGraph()
        print p
        #print s
        print


def Twostep_Fractions(showprob=False):
    
    signs = np.rint(np.random.random(3)).astype(int)
    nums = np.random.random(6) * 10 + 1
    
    nums=np.rint(nums).astype(int)    

    for i,s in enumerate(signs):
        if s==0: nums[2*i]*=-1
        
    # convert to reduced form to make things pretty
    a = Fraction(nums[0],nums[1]).numerator
    b = Fraction(nums[0],nums[1]).denominator
    c = Fraction(nums[2],nums[3]).numerator
    d = Fraction(nums[2],nums[3]).denominator
    e = Fraction(nums[4],nums[5]).numerator
    f = Fraction(nums[4],nums[5]).denominator
    
    c = abs(c)
    d = abs(d)
        
    fracs = []
    for q in [(a,b),(c,d),(e,f)]:
        if q[1]==1: fracs.append(str(q[0]))
        else: fracs.append('%u/%u' % (q[0],q[1]))
    
    if signs[1]==0: sgn = '-'
    else: sgn = '+'
    
    prob = fracs[0] + ' * x ' + sgn + ' ' + fracs[1] + ' = ' + fracs[2]
    
    #print 'Solve: %u/%u * x + %u/%u = %u/%u' % (nums[0],nums[1],nums[2],nums[3],nums[4],nums[5])
    
    F = Fraction(nums[1]*(nums[3]*nums[4]-nums[2]*nums[5]),(nums[0]*nums[3]*nums[5]))
    
    Fn = F.numerator
    Fd = F.denominator
    
    if Fd==1: soln = str(Fn)
    elif Fd==-1: soln = str(-1*Fn)
    else: soln = '%u/%u' % (Fn,Fd)

    if showprob:
        print prob
        print 
        print 'Soln ' + soln
        print 'Solution: x = %u/%u' % (F.numerator,F.denominator)    
        
    return prob,soln
    
def Log_simplify():
	
	'''
	Simplify: A*log(a) + B*log(b) + C*log(c)
	'''
	
	# get random integers for A,B,C
	A,B,C = np.random.randint(1,3,3)
	
	# get random fractions for a,b,c
	a,b,c = getFracs(3,False,False)
	
	# get signs for coefficients:
	S = getSigns(3)
	
	#write problem
	prob = S[0] + str(A) + '*log(' + str(a) + ') ' + S[1] + ' ' + str(B) + '*log(' + str(b) + ') ' + S[2] + ' ' + str(C) + '*log(' + str(c) + ')'
	if S[0]=='+': prob = prob[1:]
	
	#print prob
	
	#get solution
	pol = sign2num(S)
	x = a**(pol[0]*A) * b**(pol[1]*B) * c**(pol[2]*C)
	
	soln = 'log(' + str(x) + ')'
	
	#print soln
	
	return prob,soln
	
def Log_solve():
	'''
	Solve for x: a + b*log_c(dx + e) = f
	'''
	
	d = getFracs(1,True,False)[0]
	c = np.random.randint(1,11)
	a,e,f = np.random.randint(-10,11,3)
	
	
	b = np.random.randint(-10,11)
	if b == 0: b=1
	
	if a == 0: prob = ''
	else: prob = str(a) + ' '
	
	if abs(b)==1: prob += num2sign(b) + ' log_' + str(c) + '(' + str(d) + '*x '
	else:         prob += num2sign(b) + ' ' + str(abs(b)) + '*log_' + str(c) + '(' + str(d) + '*x '
	
	if e == 0: prob += ') = ' + str(f)
	else: prob +=  num2sign(e) + ' ' + str(abs(e)) + ') = ' + str(f)
	
	#print prob
	
	#get solution
	
	x = (c**((f-a)/b) - e)/d
	
	soln = 'x = %0.3f' % x
	
	#print soln
	
	return prob,soln
	
def MovingDecimals():
	'''
	1.514 x 10^3 = 1514
	'''
	
	r = np.random.randint(0,100) + np.random.random(1)[0]
	
	r = np.around(r,3)
	
	p = np.random.randint(-5,6)
	
	op = getSigns(1)[0]
	
	if op=='+':
		op='x'
	else:
		op='/'
		p *= -1
		
	prob = str(r) + ' ' + op + ' 10^' + str(p)
	
	s = r*10**p
	
	soln = str(s)
	
	return prob,soln
	

def Test_Problem(kind,*args):
	for i in range(10):
		print kind(*args)
	
def Homework(probs=5,days=7,fname='',kind=simpleFractions,*args):
    
    '''
    Generate problems sets.  Problem functions should return a single problem with strings ('prob','soln').
    These will be compiled into pset and answer key.
    
    INPUTS
    fname = prefix for pset filename.  Result is 'fname_pset.txt' and 'fname_answers.txt'
    kind  = problem-generating function 
    *args = arguments for kind(args)
    '''
    if fname != '': fname += '_'
    
    pset = np.empty((days,probs),dtype=object)
    
    for d in range(days):
        for p in range(probs):
            #pset[d,p] = Twostep_Fractions()                             #pset contains [problem, solution]
            pset[d,p] = kind(*args)
            
    with open(fname + 'pset.txt','w') as f:
        
        # print problems
        for d in range(days):
            f.write('Day %u:' % (d+1) + '\n')
            f.write('\n')
            for p in range(probs):
                f.write(str(p+1) + ') ' + pset[d,p][0] + '\n')
                #f.write('\n')
                
            f.write('\n')
        f.write('\n')
    
    with open(fname + 'answer_key.txt','w') as f:
        
        f.write("Answer Key:" + '\n')
        f.write('\n')
        
        for d in range(days):
            f.write('Day %u:' % (d+1) + '\n')
            f.write('\n')
            for p in range(probs):
                f.write(str(p+1) + ') ' + pset[d,p][1] + '\n')
                #f.write('\n')
            f.write('\n')
    
if __name__ == '__main__':
    Homework(probs=10,kind=MovingDecimals)
    #Test_Problem(MovingDecimals)
