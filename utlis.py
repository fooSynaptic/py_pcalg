import sys
import numpy as np
import random
from math import sqrt, log
import pandas as pd

class Matrix(object):
	def __init__(self, M):
		self.M = M
		self.shape = M.shape
		
	def diag(self, val):
		x, y = self.shape
		assert x == y
		for i in range(x):
			for j in range(y):
				if i == j:
					self.M[i,j] = val
				
	def any(self):
		x, y = self.shape
		assert x==y
		for i in range(x):
			for j in range(y):
				if self.M[i,j] == 1:
					return True
		else:
			return False

	def which(self, val):
		idx = []
		x, y = self.shape
		assert x == y
		for i in range(x):
			for j in range(y):
				if self.M[i, j] == val:
					idx.append((i,j))
		idx = np.array(idx)
		idx = idx[np.lexsort(idx[:,::-1].T)]
		return idx





#print(gen_inv(a, compute_uv = False))

#define getnextset nextSet = getNextSet(length_nbrs, ord, S)
def getNextSet(n, k, set):
	ch = list(range(n-k+1, n+1))
	leave_set = [check == 0 for check in [x-y for x,y in zip(ch, set)]]
	zeros = sum(leave_set)
	#zeros = [check for check in [x for x in ch if x not in set] if check == 0]
	#print(zeros)
	chind = k - zeros
	waslast = chind == 0
	if not waslast:
		if len(set) == 1:
			s_ch = set[0] + 1
		elif len(set) == 0 or chind-1 > len(set) or chind-1 < 0:
			return
		else:
			try:
				s_ch = set[chind-1] + 1
			except:
				#print("child",chind)
				return

		set[chind-1] = s_ch
		if chind < k:
			set[chind : k] = range(s_ch + 1, s_ch + zeros)
	
	#return $nextset, $waslast
	return {'set':set, 'waslast':waslast}


#define the indepndent test from scratch
# pval = indepTest(x, y, nbrs[S], suffStat)
# suffstat(1: cormatrix, 2:nubmer of dim)
def indTest(x, y, S, suffStat):
	z = zstat(x, y, S, suffStat[0], suffStat[1])
	print("we got the indepedent statistical val of :\t", z)
	cnt = 0
	#cset = 1 * np.random.random_sample((10000)) - 1
	p = len([x for x in range(100000) if x<z])/100000
	if p < 0.00001:
		return 0
	else:
		return p


# zStat() gives a number
# Z = sqrt(n - |S| - 3) * log((1+r)/(1-r))/2
def zstat(x, y, S, C, n):
	try:
		assert isinstance(S, list)
	except:
		S = [S]
	print(S)
	r = pcorOrder(x,y,S[0],C)
	res = sqrt(n - len(S) - 3)*0.5*log((1+r)/(1-r))/2
	if not res:
		return 0
	else:
		return res


# compute partial corrlations

def pcorOrder(i,j,k,C, cut = 0.99999):
	k = [k]
	if len(k) == 0:
		r = C[i,j]
	elif len(k) == 1:
		idx = k[0]
		r = (C[j][i] - C[idx][i]*C[idx][j])/sqrt((1 - C[idx][j]**2)*(1 - C[idx][i]**2))
	else:
		mat = C[[i,j,k]].iloc[[i,j,k],:]
		_pm = pseudoinverse(mat)
		r = - _pm[2][1]/sqrt(_pm[1][1]*_pm[2][2])
	#print(type(r), r.values())
	if not r:
		return 0
	else:
		return min(cut, max(-cut, r))

	

def pseudoinverse(m):
	# we need the module preform the svd
	msvd = gen_inv(m)
	pos_vec = [x for x in msvd[1] if x>0]
	if len(pos_vec) == 0:
		return np.zeros((m.shape[::-1]))
	else:
		return np.dot(msvd[2], (np.array([1/x for x in pos_vec]) * msvd[0].T))



#define the singular value decomposition


gen_inv = np.linalg.svd