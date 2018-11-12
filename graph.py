import networkx as nx
import matplotlib.pyplot as plt

'''
#G = nx.Graph()#创建空的网络图
#G = nx.DiGraph()
#G = nx.MultiGraph()
#G = nx.MultiDiGraph()


FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
for n, nbrs in FG.adj.items():
   for nbr, eattr in nbrs.items():
       wt = eattr['weight']
       if wt < 0.5: print('(%d, %d, %.3f)' % (n, nbr, wt))


nx.draw(FG, with_labels=True, font_weight='bold')
#nx.draw(G,pos = nx.random_layout(G)，node_color = 'b',edge_color = 'r',with_labels = True，font_size =18,node_size =20)

plt.show()
'''



def grapher(netmatrix, label_dict, *kwargs):
	network = nx.Graph()
	res = []
	shape = netmatrix.shape
	for i in range(shape[0]):
		for j in range(shape[1]):
			val = netmatrix.M[i,j]
			if val>0:
				res.append((label_dict[i], label_dict[j], val))
	print(res)	
	network.add_weighted_edges_from(res)


	for n, nbrs in network.adj.items():
		for nbr, eattr in nbrs.items():
			wt = eattr['weight']
			if wt < 0.5: print('(%d, %d, %.3f)' % (n, nbr, wt))
	
	nx.draw(network, with_labels=True, font_weight='bold')

	plt.show()

	

