import sys
import numpy as np



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

a = np.ones((5,5))

a = Matrix(a)
print(a.M)
print(a.which(1))



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
				nbrs = seq_p
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