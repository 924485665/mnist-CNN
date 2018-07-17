import networkx as nx
import matplotlib.pyplot as plt
import numpy
from numpy import *
import time
# import copy

def fill_adjacent_matrix(E):                                     #填充邻接矩阵函数
    for i in range(len(E)):
        W[E[i][0]-1][E[i][1]-1] = 1

def dis(a,b):                                                       #计算节点欧氏距离
    vec1 =nodes_positions[a]
    vec2 =nodes_positions[b]

    return numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))

def update_Rmatrix(A,R):
    Rold = copy.copy(R)
    Rold = copy(R)
    Rnew = copy.copy(R)
    for i in range(G.number_of_nodes()):
        for j in range(G.number_of_nodes()):
            if(j!=0):                                                #赋一个临时的初值
                temp = A[i][j-1] + S[i][j-1]
            else:
                temp = A[i][j+1] + S[i][j+1]
            for k in range(G.number_of_nodes()):
                if(k != j):
                    temp = max(temp,A[i][k] + S[i][k])
            Rnew[i][j] = S[i][j] - temp
    R = damp2*Rnew + (1-damp2)*Rold
    return R




def update_Amatrix(A,R):
    Aold = copy.copy(A)                                #浅拷贝
    Anew = copy.copy(A)                                #浅拷贝
    # print("Anew = ")
    # print(Anew)
    # print("Aold = ")
    # print(Aold)
    for i in range(G.number_of_nodes()):
        for j in range(G.number_of_nodes()):
            if(i==j):
                sum = 0
                for k in range(G.number_of_nodes()):
                    if(k != j):
                        sum += max(0,R[k][j])
                A[i][j] = sum
            else:
                sum = 0
                for k in range(G.number_of_nodes()):
                    if(k != j and k != i):
                        sum += max(0,R[k][j])
                Anew[i][j] = min(0,R[j][j]+sum)
    print("Anew = ")
    print(Anew)
    print("Aold = ")
    print(Aold)
    A = damp2 * Anew + (1 - damp2) * Aold
    return A

def calculate_centers(A,R):
    for i in range(G.number_of_nodes()):
        centers.append({'num_nodes':i+1,'value':A[i][i]+R[i][i]})
    print(centers)
    centers.sort(key = lambda x:x['value'],reverse = True)
    print(centers)


G = nx.read_gml('E:\社区 数据\karate\karate.gml')                    #karate数据集
W = zeros([G.number_of_nodes(),G.number_of_nodes()],dtype = int)     #初始化结点的邻接矩阵
fill_adjacent_matrix(G.edges())                                      #填充邻接矩阵
S = zeros([G.number_of_nodes(),G.number_of_nodes()],dtype = double)     #初始化相似度矩阵
centers = list()                                                     #初始化一个存储聚类中心的列表
damp1 = 0.9
damp2 = 0.9

path=nx.all_pairs_shortest_path(G)                            #调用多源最短路径算法，计算图G节点间的最短路径
nx.draw_networkx(G)
nodes_positions = {}
nodes_positions = nx.drawing.spring_layout(G).copy()          #节点位置词典，通过Fruchterman-Reingold算法排列节点
# print(nodes_positions)


for i in range(G.number_of_nodes()):                            #计算出相似度矩阵，欧氏距离及节点间距离加权和的赋值
    for j in range(G.number_of_nodes()):
        S[i][j] = -damp1*(len(path[i+1][j+1])-1)+ (1-damp1)*dis(i+1,j+1)

s = 0.0
for i in range(G.number_of_nodes()):                            #求出相似度矩阵的中位数作为对角线元素
    for j in range(G.number_of_nodes()):
        if(i!=j):
            s +=S[i][j]
mean = s/(G.number_of_nodes()*(G.number_of_nodes()-1))
for i in range(G.number_of_nodes()):
    S[i][i] = mean
print('S =')
print(S)


for i in range(G.number_of_nodes()):                             #将中位数赋值给S[u][u]
    S[i][i] = mean



A = zeros([G.number_of_nodes(),G.number_of_nodes()],dtype = double)
R = zeros([G.number_of_nodes(),G.number_of_nodes()],dtype = double)
iter = 20
for i in range(10):
    print("iter = %d" %(i+1))
    print()
    R1 = R
    R = update_Rmatrix(A,R)
    print('R =')
    print(R)
    A = update_Amatrix(A,R1)
    print('A =')
    print(A)

calculate_centers(A,R)                               #生成排好序的聚类中心矩阵


#分配社区 （根据基于合并社区边缘孤立节点的方法来）
print(centers[0]['num_nodes'])
print(centers[1]['num_nodes'])
belongs = zeros(G.number_of_nodes(),dtype = int)
first_com_nodes = list()
second_com_nodes = list()
belongs[centers[0]['num_nodes']-1] = 1
belongs[centers[1]['num_nodes']-1] = 2


#合并社区第一步，将与边缘节点直接相连的边加入他们对应的社区
for i in range(G.number_of_edges()):
    if (centers[0]['num_nodes'] == G.edges()[i][0] or centers[0]['num_nodes'] == G.edges()[i][1]):
        if (centers[0]['num_nodes'] == G.edges()[i][0]):
            first_com_nodes.append(G.edges()[i][1])
            belongs[G.edges()[i][1]-1] = 1
        else:
            first_com_nodes.append(G.edges()[i][0])
            belongs[G.edges()[i][0]-1] = 1
    else:
        if (centers[1]['num_nodes'] == G.edges()[i][0] or centers[1]['num_nodes'] == G.edges()[i][1]):
            if (centers[1]['num_nodes'] == G.edges()[i][0]):
                second_com_nodes.append(G.edges()[i][1])
                belongs[G.edges()[i][1]-1] = 2
            else:
                second_com_nodes.append(G.edges()[i][0])
                belongs[G.edges()[i][0]-1] = 2

print(first_com_nodes)
print(second_com_nodes)
print(belongs)