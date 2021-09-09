import networkx as nx
import numpy as np
g=nx.random_graphs.watts_strogatz_graph(6,3,0.3)
colors = [1, 2, 3, 4, 5, 6]
nx.draw_kamada_kawai(g,with_labels=True,node_color=colors,alpha=0.7,node_size=600,font_size =18)
A=nx.adjacency_matrix(g).todense()
print(A)
X = np.matrix([
            [i, -i]
            for i in range(A.shape[0])
        ], dtype=float)
print(X)
AX = A*X
print(AX)

I=np.matrix(np.eye(A.shape[0]))  #单位矩阵
print(I)
A_hat = A+I
print("A_hat")
print(A_hat)
middle = A_hat.sum(axis=0)
middle1 = np.diag(np.array(middle)[0])
D=np.matrix(np.diag(np.array(A_hat.sum(axis=0))[0]))
print("D")
print(D)
print("D_reverse")
print(D**-1)
D_minus1_AX = D**-1*A_hat
print("D_minus1_AX")
print(D_minus1_AX)