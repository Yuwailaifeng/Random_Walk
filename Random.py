#!/usr/local/bin/python3
#-*- coding: utf-8 -*-


import numpy as np
import random

def PersonalRank(G, walk, root, max_step):
	PR = dict()
	PR = { node:0 for node in G.keys() }
	PR[root] = 1
	print(PR)

	#开始迭代
	for k in range(max_step):
		tmp = { node:0 for node in G.keys() }

		#取节点i和它的出边尾节点集合out
		for j, out in G.items():

			#取节点i的出边的尾节点j以及边E(i,j)的权重wij, 边的权重都为1，在这不起实际作用
			for i, wij in out.items():

				#i是j的其中一条入边的首节点，因此需要遍历图找到j的入边的首节点，
				#这个遍历过程就是此处的2层for循环，一次遍历就是一次游走
				tmp[i] += walk * PR[j] / (1.0 * len(out))
				
		#每次游走都是从root节点出发，因此root节点的权重需要加上(1 - walk)
		tmp[root] += (1 - walk)
		PR = tmp

		#输出每次迭代后各个节点的权重
		print('iter: ' + str(k) + "\t", end='')
		for key, value in PR.items():
			print("%s: %.6f \t"%(key, value), end='')
		print()

	return PR



def RandomWalk(G, root, steps):

	# nodes = ['A', 'B', 'C', 'a', 'b', 'c', 'd']

	# adjacency_matrix = [
	# 	  [0, 0, 0, 1, 0, 1, 0],  # A
	#	 [0, 0, 0, 1, 1, 1, 1],  # B
	#	 [0, 0, 0, 0, 0, 1, 1],  # C
	#	 [1, 1, 0, 0, 0, 0, 0],  # a
	#	 [0, 1, 0, 0, 0, 0, 0],  # b
	#	 [1, 1, 1, 0, 0, 0, 0],  # c
	#	 [0, 1, 1, 0, 0, 0, 0]]  # d


	nodes = [ k for k in G.keys() ]

	adjacency_matrix = []
	for k,v in G.items():
		temp = [ 1 if nodes[i] in v.keys() else 0 for i in range(len(nodes)) ]
		adjacency_matrix.append(temp)

	degree = [ sum(i) for i in adjacency_matrix ]
	A = np.matrix(adjacency_matrix)
	D = np.matrix( np.diag(degree) )
	L = np.matrix( D - A )
	print(nodes)
	print(A)
	print(D)
	print(L)


	# P3方法
	P = D.I * A
	# print( np.around( P, decimals=6) )
	print()
	print( np.around( P ** steps, decimals=6) )


	# P3方法 矩阵分块
	P = np.array(D.I * A)
	# print( np.around( P, decimals=6) )
	UI = np.matrix( P[:3, 3:] )
	IU = np.matrix( P[3:, :3] )
	print()
	print( np.around( UI * np.power(IU*UI, (steps-1)/2), decimals=6) )


	# Pa3方法
	a = 1.6
	P = np.power((D.I * A), a)
	# print( np.around( P, decimals=6) )
	print()
	print( np.around( P ** steps, decimals=6) )


	# RPb3方法
	b = 0.3
	P = D.I * A
	# print( np.around( P/b, decimals=6) )
	print()
	print( np.around( (P ** steps) / np.power(degree, b), decimals=6) )


	# RPb3方法  采样
	UI = np.matrix( A[:3, 3:] ).tolist()
	IU = np.matrix( A[3:, :3] ).tolist()
	RP = np.matrix( np.zeros( (len(UI),len(IU)) ) ).tolist()

	for c in range(1000):

		u = random.randint(0, len(UI)-1)

		t1 = [ k for k in range(len(IU)) if UI[u][k] == 1 ]
		j = random.choice(t1)

		t2 = [ k for k in range(len(UI)) if IU[j][k] == 1 ]
		v = random.choice(t2)

		t3 = [ k for k in range(len(IU)) if UI[v][k] == 1 ]
		i = random.choice(t3)

		pui = UI[u][j] * IU[j][v] * UI[v][i]
		crw = np.power( degree[u]*degree[j+3]*degree[v], 0.7 )
		RP[u][i] += pui*crw
	
	print()
	print( np.around( np.matrix(RP)/np.matrix(RP).sum(axis=1), decimals=6) )




	# # RPb3方法  采样
	# UI = np.matrix( A[:3, 3:] ).tolist()
	# IU = np.matrix( A[3:, :3] ).tolist()
	# RP = np.matrix( np.zeros( (len(UI),len(IU)) ) ).tolist()
	# for c in range(10000):
	# 	u = random.randint(0, len(UI)-1)
	# 	j = random.randint(0, len(IU)-1)
	# 	v = random.randint(0, len(UI)-1)
	# 	i = random.randint(0, len(IU)-1)
	# 	pui = UI[u][j] * IU[j][v] * UI[v][i]
	# 	crw = np.power( degree[u]*degree[j+3]*degree[v], 0.7 )
	# 	RP[u][i] += pui*crw
	# print()
	# print( np.around( np.matrix(RP), decimals=6) )



	# # 输出推荐列表
	# R = ( (P ** steps)/np.power(degree, b) ).tolist()
	# RR = [ nodes[i] + ': ' + str(R[ nodes.index(root) ][i]) for i in range(3,len(nodes)) if adjacency_matrix[ nodes.index(root) ][i] == 0 ]
	# print(root + '=', end='')
	# print(RR)
	# print()








if __name__ == '__main__' :

	G = {'A' : {'a' : 1, 'c' : 1},
		 'B' : {'a' : 1, 'b' : 1, 'c':1, 'd':1},
		 'C' : {'c' : 1, 'd' : 1},
		 'a' : {'A' : 1, 'B' : 1},
		 'b' : {'B' : 1},
		 'c' : {'A' : 1, 'B' : 1, 'C':1},
		 'd' : {'B' : 1, 'C' : 1}}

	# PersonalRank(G, 0.85, 'A', 200)
	RandomWalk(G, 'A', 3)






