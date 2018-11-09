import sys
import numpy as np
import random
from math import sqrt, log
import pandas as pd
 

#set random vector
cset = 1 * np.random.random_sample((10000)) - 1

class Matrix(object):
	def __init__(self, M):
		self.M = M
		self.shape = M.shape
		
	def diag(self, val):
		x, y = self.shape
		assert x == y
		for i in range(x):
			for j in range(y):
				self.M[x,y] == val
				
	def any(self):
		x, y = self.shape
		assert x==y
		for i in range(x):
			for j in range(y):
				if self.M[i,j] == 1:
					return True
		else:
			return False
	
	def numeric(self, n):
		return tuple([0]*n)

	def which(self, val):
		idx = []
		x, y = self.shape
		assert x == y
		for i in range(x):
			for j in range(y):
				if self.M[i, j] == val:
					idx.append((i,j))
		return idx

'''
a = np.ones((5,5))

a = Matrix(a)
print(a.M)
print(a.which(1))
'''

#define getnextset nextSet = getNextSet(length_nbrs, ord, S)
def getNextSet(n, k, set):
	ch = list(range(n-k+1, n+1))
	zeros = [check for check in [x for x in ch if x not in set] if check == 0]
	chind = k - zeros
	waslast = chind == 0
	if not waslast:
		s_ch = set[chind] + 1
		set[chind] = s_ch
		if chind < k:
			set[chind+1 : k] = range(s_ch + 1, s_ch + zeros)
	
	#return $nextset, $waslast
	return set, waslast



#define the indepndent test from scratch
# pval = indepTest(x, y, nbrs[S], suffStat)
# suffstat(1: cormatrix, 2:nubmer of dim)
def Test(x, y, S, suffStat):
	z = zstat(x, y, S, suffStat[0], suffStat[1])
	print("we got the indepedent statistical val of :\t", z)
	cnt = 0
	#cset = 1 * np.random.random_sample((10000)) - 1
	p = len([x for x in range(100000) if x<z])/100000
	return p


# zStat() gives a number
# Z = sqrt(n - |S| - 3) * log((1+r)/(1-r))/2
def zstat(x, y, S, C, n):
	try:
		assert isinstance(S, list)
	except:
		S = [S]

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
		r = (C[i][j] - C[i][idx]*C[j][idx])/sqrt((1 - C[j][idx]**2)*(1 - C[i][idx]**2))
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

#test
def debug_trivial():
	a = np.random.randn(50, 50)
	rev = pseudoinverse(a)
	print('rev eigen', rev)
	#correlation matrix
	dfa = pd.DataFrame(a).corr()
	pcor = pcorOrder(1,2,3,dfa)
	print("partial correlation", pcor)


	zs = zstat(1, 2, 3, dfa, 50)
	print("z statitical", zs)

	print("final test:", Test(1,2,3, [dfa, 50]))
#print(gen_inv(a, compute_uv = False))





def skeletion(suffStat, indepTest, alpha, labels, method, fixedGaps=None, fixdEdges=None, NAdelete=True,\
m_max = float('Inf'), u2pd=("relaxed","rand","retry"),solve_confl = False, numCores = 1, verbose = False):
	try:
		l, p = labels, p
	except:
		raise Exception("Argument need to specify 'labels' or 'p'!")

	seq_p = list(range(p))
	if not fixedGaps:
		G = np.ones(shape = (p, p))
	elif fixedGaps.shape != (p,p):
		raise Exception("dimensitons of the dataset and fixedGaps do not agree.")
	elif fixedGaps.shape != fixedGaps.T.shape:
		raise Exception("fixedGaps must be symmetric.")
	else:
		G = np.zeros(shape = (p, p))
	for i in range(G.shape[0]):
		for j in range(G.shape[1]):
			if i == j:
				G[i,j] == 0
	
	#initial the fixedEdges if it's none
	if not fixedEdges:
		fixdEdges = np.zeros(shape = (p,p))

	'''
	if method == "stable.fast":
		indepTestName = 'gauss'
	else:
		indepTestName = 'rfun'
	'''

	#inference
	pval = None
	sepset = [[None]*p for i in range(p)]
	pMax = np.matrix([float('Inf') for i in range(p)])
	pMax = diag(pMax, 1)
	done = False
	ord = 0

	n_edgetests = numeric(1)
	while not done and any(G) and ord <= m_max:
		ord1 = ord + 1
		n_edgetests[ord1] = 0
		done = True
		ind = G.which(1)
		remEdges = G.shape[1]

		for i in range(remEdges):
			x = ind[i, 0]
			y = ind[i, 1]

			if G.M[y,x] and not fixedEdges[y, x]:
				nbrsBool = G.M[:, x]
				nbrsBool[y] = 0
				#merge seq_p && nbrsBool
				nbrs = [seq_p[i] for i in range(len(seq_p)) if seq_p[i] and nbrsBool[i]]
				length_nbrs = len(nbrs)
				if length_nbrs >= ord:
					if length_nbrs > ord:
						done = False
					S = list(range(ord))
					while(1):
						n_edgetests[ord1] = n_edgetests[ord1] + 1
						pval = indepTest(x, y, nbrs[S], suffStat)
						if pval == None:
							pval = int(NAdelete)
						if pMax[x,y] < pval:
							pMax[x,y] = pval
						if pval >= alpha:
							G[x,y] = G[y,x] = 0
							sepset[x][y] = nbrs[S]
							break
						else:
							nextSet = getNextSet(length_nbrs, ord, S)


		pass








	pass
	



def pc(suffStat, indepTest, alpha, labels, p, fixedGaps=None, fixdEdges=None, NAdelete=True,\
m_max = float('Inf'), u2pd=("relaxed","rand","retry"), skel_method = ('stable','original'),\
solve_confl = False, numCores = 1, verbose = False):
	try:
		l, p = labels, p
	except:
		raise Exception("Argument need to specify 'labels' or 'p'!")

	
	pass

#pc()
