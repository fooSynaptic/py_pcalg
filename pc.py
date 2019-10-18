import sys
import numpy as np
import random
from math import sqrt, log
import pandas as pd
from utlis import *
from graph import grapher

#set random vector
cset = 1 * np.random.random_sample((10000)) - 1



def skeletion(suffStat, indepTest, alpha, labels, fixedGaps=None, fixedEdges=None, NAdelete=True,\
m_max = float('Inf'), u2pd=("relaxed","rand","retry"),solve_confl = False, numCores = 1, verbose = False):
	try:
		l = labels
		p = len(labels)
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
	'''
	for i in range(G.shape[0]):
		for j in range(G.shape[1]):
			if i == j:
				G[i,j] == 0
	'''
	#class G as a matrix class
	G = Matrix(G)
	#set diag
	G.diag(0)
	
	#initial the fixedEdges if it's none
	if not fixedEdges:
		fixedEdges = np.zeros(shape = (p,p))

	'''
	if method == "stable.fast":
		indepTestName = 'gauss'
	else:
		indepTestName = 'rfun'
	'''

	#inference
	pval = None
	sepset = [[None]*p for i in range(p)]
	pMax = np.matrix([float('Inf') for i in range(p*p)]).reshape(p, -1)
	pMax = Matrix(pMax)
	pMax.diag(1)
	done = False
	ord = 0

	n_edgetests = [0]*p
	while not done and G.any() and ord <= m_max:
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
					S = list(range(ord+1))
					if len(S) == 0:
						return G
					
					while True:
						n_edgetests[ord1] = n_edgetests[ord1] + 1
						try:
							pval = indepTest(x, y, [nbrs[x] for x in S if x <= len(nbrs)], suffStat)
						except:
							print(S, nbrs)
						if not pval:
							pval = int(NAdelete)
						if pMax.M[x,y] < pval:
							pMax[x,y] = pval
						elif pval >= alpha:
							G.M[x,y] = G.M[y,x] = 0
							try:
								sepset[x][y] = [nbrs[x] for x in S]
							except:
								return G
								
							break
						else:
							nextSet = getNextSet(length_nbrs, ord, S)
						if nextSet['waslast']:
							break
						S = nextSet['set']
						# end for loop
		ord = ord + 1
		#end while loop
	for i in range(1, p):
		for j in range(2, p):
			pMax.M[i,j] = pMax.M[j,i] = max(pMax.M[i,j], pMax.M[j,i])


	return G


def pc(suffStat, indepTest, alpha, labels, p, fixedGaps=None, fixdEdges=None, NAdelete=True,\
m_max = float('Inf'), u2pd=("relaxed","rand","retry"), skel_method = ('stable','original'),\
solve_confl = False, numCores = 1, verbose = False):
	try:
		l, p = labels, p
	except:
		raise Exception("Argument need to specify 'labels' or 'p'!")
#pc()


def debug_trivial():
	'''
	a = np.random.randn(50, 50)
	'''

	b = pd.read_csv('./test_data.csv')
	data = np.array(b.iloc[:,:])[:,1:]
	a = pd.DataFrame(data).corr()
	#names =["space","中","体育讯","分","新浪","比赛","球员","球队","日","月","北京","时间"]
	names = ['space', 'middle', 'Sports', 'min', 'sina', 'match', 'player', 'team', 'day', 'month', 'peking', 'time']
	assert data.shape[1]==len(names)
	label_dict = dict(zip(list(range(data.shape[1])), names))

	rev = pseudoinverse(a)
	print('rev eigen', rev)
	#correlation matrix
	dfa = pd.DataFrame(a).corr()
	pcor = pcorOrder(1,2,3,dfa)
	print("partial correlation", pcor)


	zs = zstat(1, 2, 3, dfa, a.shape[1])
	print("z statitical", zs)

	print("final test:", indTest(1,2,3, [dfa, a.shape[1]]))
	print("Test the next independent set:\n",'***'*5)
	next = getNextSet(5, 2, [1,2])
	print(next, "refer: [1,3], False")
	next = getNextSet(5, 2, [4,5])
	print(next, "refer: [4,5], True")

	grap = skeletion([dfa, a.shape[1]], indTest, alpha = 0.05, labels = list(range(a.shape[1])), fixedGaps=None, fixedEdges=None, NAdelete=True,\
m_max = float('Inf'), u2pd=("relaxed","rand","retry"),solve_confl = False, numCores = 1, verbose = False)
	print(grap.M, grap.shape)
	#return

	grapher(grap, label_dict)



if __name__ == "__main__":
	debug_trivial()
	#pass