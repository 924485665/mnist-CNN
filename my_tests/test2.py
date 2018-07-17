#-*- coding:utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
# G = nx.read_gml('E:\社区 数据\karate\karate.gml')
# print(G.nodes())
# print(G.edges())
# print(G.edges()[2][0])
# print(G.edges()[2][1])
# print(G.number_of_nodes())
# print(G.number_of_edges())
# nx.draw(G)
# nx.draw_networkx(G)
# plt.show()
# G2 = nx.read_gml('E:\社区 数据\dolphins\dolphins.gml')
# nx.draw_networkx(G2)
# plt.show()
# # plt.savefig("E:/社区 数据/my_nodes_pics/"+time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))+"network.png")   #存储该图片
# # plt.savefig("C:/network.png")
# nodes_positions ={}
# nodes_positions = nx.drawing.spring_layout(G).copy()
# print(nodes_positions)
# print('!!!')
# print(nodes_positions[1])
# print(nodes_positions[1][0])
# print()
#
# # print(nx.node_collection)
# plt.show()
#
# # G1 = nx.random_graphs.barabasi_albert_graph(100,1)
# # nx.draw(G1)
# # plt.show()
# path=nx.all_pairs_shortest_path(G)     #调用多源最短路径算法，计算图G所有节点间的最短路径
# # print(path[17][9])                                   #输出节点之间的最短路径序列
# # print(path[27][9])
# # print(path[17][3])
# # print(path[27][3])
#
#
# # l1 = [1,2,3,4,5,6,1]
# # if (1 in l1 and 2 in l1):
# #     print('True')
# # else:
# #     print('False')
#
# print(G.edges())
# for i in range(G.number_of_edges()):
#     print("the edges of %dth node" %(i+1))
#     for k in range(G.number_of_edges()):
#         if((i+1) == G.edges()[k][0] or (i+1) == G.edges()[k][1]):
#             print(G.edges()[k])
#
# import numpy as np
# # from numpy import *
# import copy
# R = np.zeros([10,10],dtype = 'int')
# # Rold = copy.copy(R)
# Rold = copy.copy(R)
# print(R)
# print(Rold)
# Rold[1][2] = 1.0
# print(R)
# print(Rold)

import numpy as np
import  math,copy
import matplotlib.pyplot as plt
import networkx as nx
import community as com


def fill_adjacent_matrix(E):                                     #填充邻接矩阵函数
    for i in range(len(E)):
        W[E[i][0]-1][E[i][1]-1] = 1



def calculate_Modulartiy(G,coms):       #求划分结果的模块度Q   (coms为列表，每个元素为一个社区的列表)

    m = len(G.edges())                         #m 为图中的总边数
    Q = 0
    for com_c in coms:
        lc,Oc,Dc = 0,0,0  # lc为社区c内部边的条数; Oc为社区c与其他社区之间的边数; Dc为社区c中所有顶点的度之和
        for each_edge in range(m):
            if (G.edges()[each_edge][0] in com_c and G.edges()[each_edge][1] in com_c):
                lc += 1
            else:
                if ((G.edges()[each_edge][0] in com_c and G.edges()[each_edge][1] not in com_c) or (
                        G.edges()[each_edge][0] not in com_c and G.edges()[each_edge][1] in com_c)):
                    Oc += 1
        Dc = 2 * lc + Oc
        Q += (lc/m - (Dc/(2*m))*(Dc/(2*m)))
    return Q

def calculate_NMI(coms1,coms2):               #计算两种社团划分的NMI值，当有的社团特别小(且分错)时，NMI会非常小   如 ([1-9],[10])和([1-8],[9,10])
    sums = sum([len(x) for x in coms1])
    N = np.zeros([len(coms1),len(coms2)],dtype = int)              #初始化混淆矩阵N，Nij表示两种社团划分的相同节点个数
    for i in range(len(N)):                                       #填充N
        for j in range(len(N[0])):
            N[i][j] = get_sums_of_samenodes(coms1,coms2,i,j)
    # NMI = -2*sum([N[i][j]*math.log((),math.e) for i in range(len(N)) for j in range(len(N[0]))])/
    A = 0.0
    # right = math.log(N[i][j] * sums / float(Ni * Nj), math.e)
    A = -2*sum([N[i][j]*math.log((N[i][j]*sums)/float(sum([x for x in N[i]])*sum([N[i][j] for i in range(len(N))])),2)  if N[i][j]!=0 else 0  for i in range(len(N)) for j in range(len(N[0]))])
    Xi,Xj = sum([sum([x for x in N[i]])*math.log(sum([x for x in N[i]])/sums,2) if sum([x for x in N[i]])!=0 else 0 for i in range(len(N))]),sum([sum([N[i][j] for i in range(len(N))])*math.log(sum([N[i][j] for i in range(len(N))])/sums,2) if sum([N[i][j] for i in range(len(N))])!=0 else 0  for j in range(len(N[0]))])
    # print(A,Xi,Xj)
    return float(A)/(Xi+Xj)



def get_sums_of_samenodes(coms1,coms2,i,j):
    A,B = set(coms1[i]),set(coms2[j])
    return len(A&B)




def calculate_jaccard(coms1,coms2):
    S1,S2 =set(),set()
    for com_c in coms1:
        for i in range(len(com_c)):
            for j in range(i+1,len(com_c)):
                S1.add((i,j))

    for com_c in coms2:
        for i in range(len(com_c)):
            for j in range(i+1,len(com_c)):
                S2.add((i,j))

    return len(S1&S2)/float(len(S1|S2))

print('jaccard = %.10f' %calculate_jaccard([[1,2,3],[4]],[[4],[3,2,1]]))
print('jaccard = %.10f' %calculate_jaccard([list(range(1,20)),[20]],[list(range(1,19)),[19,20]]))







# print(calculate_NMI([[1,2],[4,3]],[[3,4],[2,1]]))
# print(calculate_NMI([[1,2,3],[4]],[[4],[3,2,1]]))
# print(calculate_NMI([list(range(1,10)),list(range(10,21))],[list(range(1,9)),list(range(9,21))]))
# print(calculate_NMI([list(range(1,10)),[10]],[list(range(1,8)),[8,9,10]]))


G_temp = nx.read_gml('E:/社区 数据/karate/karate.gml')
#将（karate数据集）的节点改为从1开始
G1 = nx.Graph()
for x in G_temp.edges():
    G1.add_edge(x[0]-1,x[1]-1)
G2 = nx.read_gml('E:/社区 数据/dolphins/dolphins.gml')
G3 = nx.read_gml('E:/社区 数据/football/football.gml')
G4 = nx.Graph()
Gtest_coms = {}
with open('E:/社区 数据/LFR datasets/benchmark -N 200 -k 20 -maxk 50 -mu 0.1 -benchmark -N 200 -k 20 -maxk 50 -mu 0.1 -/mu0.1/network.dat','r') as f:
    for x in f:
        G4.add_edge(int(x.strip().split()[0])-1,int(x.strip().split()[1])-1)

with open('E:/社区 数据/LFR datasets/benchmark -N 200 -k 20 -maxk 50 -mu 0.1 -benchmark -N 200 -k 20 -maxk 50 -mu 0.1 -/mu0.1/community.dat','r') as f:
    for x in f:
        node,community = int(x.strip().split()[0])-1,int(x.strip().split()[1])-1
        if Gtest_coms.get(community)==None:
            Gtest_coms[community] = [node]
        else:
            Gtest_coms[community].append(node)
Gtest_coms = list(Gtest_coms.values())


# W = np.zeros([G.number_of_nodes(),G.number_of_nodes()],dtype = 'int')     #初始化结点的邻接矩阵
# fill_adjacent_matrix(G.edges())                                      #填充邻接矩阵
# print(G.edges())
# print(W)
# S = np.zeros([G.number_of_nodes(),G.number_of_nodes()],dtype = 'double')     #初始化相似度矩阵
# centers = list()                                                     #初始化一个存储聚类中心的列表
# damp1 = 0.9
# damp2 = 0.9
#
# path=nx.all_pairs_shortest_path(G)                            #调用多源最短路径算法，计算图G节点间的最短路径
# nx.draw_networkx(G)
# print()
# nodes_positions = {}
# nodes_positions = nx.drawing.spring_layout(G).copy()          #节点位置词典，通过Fruchterman-Reingold算法排列节点
# # print(nodes_positions)


#利用Fast unfolding 方法的高 模块度Q
def fast_unfolding(G,reference):
    partition = com.best_partition(G)
    # print(partition)
    # print(partition.values())
    # print(partition.keys())
    com_dict = {}
    for key in partition:
        if com_dict.get(partition[key]) == None:
            com_dict[partition[key]] = [key]
        else:
            com_dict[partition[key]].append(key)

    # print(com_dict)
    result_coms = [com_dict[key] for key in com_dict]
    reference = result_coms
    Q = calculate_Modulartiy(G, result_coms)
    NMI = calculate_NMI(result_coms,reference)
    jaccard = calculate_jaccard(result_coms,reference)
    return Q,NMI,jaccard,result_coms

def LPA(G,reference):
    #分配标签,用labels字典存储
    labels = dict((i,i) for i in G.nodes())
    t,flag = 1,True
    while t<=100 and flag == True:
        flag = False
        for x in G.nodes():
            counts = np.bincount([labels[key] for key in G.neighbors(x)]).tolist()  #统计当前节点邻居节点的各标签个数
            max_pos = np.argmax(counts)   #记录最大标签数的索引
            same_list = []                #存放可能的多个最大标签数
            for i, j in enumerate(counts):
                if j == counts[max_pos]:
                    same_list.append(i)
            if labels[x] not in same_list:
                labels[x] = random.sample(same_list,1)[0]
                flag = True
        t +=1
    print('t={}'.format(t))
    result_coms1 = labels
    my_d = {}
    for key in result_coms1:
        if my_d.get(result_coms1[key]) == None:
            my_d[result_coms1[key]] = [key]
        else:
            my_d[result_coms1[key]].append(key)
    result_coms = list(my_d.values())
    Q = calculate_Modulartiy(G,result_coms)
    NMI = calculate_NMI(result_coms,reference)
    jaccard = calculate_jaccard(result_coms,reference)
    return Q,NMI,jaccard,result_coms


#根据标签熵来得到当前的节点更新序列
def calculate_H(G,labels):
    nodes_H = []
    for node in G.nodes():
        counts = np.bincount([labels[key] for key in G.neighbors(node)]).tolist()  #统计当前节点邻居节点的各标签个数
        neis = {}
        for i,j in enumerate(counts):
            if j!=0:
                neis[i] = j
        if neis.get(node)==None:
            neis[node] = 1
        else:
            neis[node] +=1
        sum_of_labels = sum(list(neis.values()))
        Pl = [x/float(sum_of_labels) for x in list(neis.values())]
        nodes_H.append({node:-sum([x*math.log(x,math.e) for x in Pl])})
    nodes_H.sort(key = lambda x:list(x.values())[0])
    return [list(x.keys())[0] for x in nodes_H]

# print(math.log(3,math.e))
# print(0.8*math.log(0.4,math.e)+0.2*math.log(0.2,math.e))

# nodes_H = [{1:0.1},{2:0.5},{3:0.3},{10:0.05},{5:0.4}]
# nodes_H.sort(key = lambda x:list(x.values())[0])
# print([list(x.keys())[0] for x in nodes_H])


# 结合标签熵的LPA
def LPAH(G,reference):
    #分配标签,用labels字典存储
    labels = dict((i,i) for i in G.nodes())
    t,flag = 1,True
    H = calculate_H(G, labels)
    while t<=100 and flag == True:
        flag = False
        for x in H:
            counts = np.bincount([labels[key] for key in G.neighbors(x)]).tolist()  #统计当前节点邻居节点的各标签个数
            max_pos = np.argmax(counts)   #记录最大标签数的索引
            same_list = []                #存放可能的多个最大标签数
            for i, j in enumerate(counts):
                if j == counts[max_pos]:
                    same_list.append(i)
            if labels[x] not in same_list:
                labels[x] = random.sample(same_list,1)[0]
                flag = True
        t +=1
    print('t={}'.format(t))
    result_coms1 = labels
    my_d = {}
    for key in result_coms1:
        if my_d.get(result_coms1[key]) == None:
            my_d[result_coms1[key]] = [key]
        else:
            my_d[result_coms1[key]].append(key)
    result_coms = list(my_d.values())
    Q = calculate_Modulartiy(G, result_coms)
    NMI = calculate_NMI(result_coms, reference)
    jaccard = calculate_jaccard(result_coms, reference)
    return Q, NMI, jaccard, result_coms



def dis(nodes_positions,a,b):                                                       #计算节点欧氏距离
    vec1 =nodes_positions[a]
    vec2 =nodes_positions[b]
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def update_Rmatrix(G,S,A,R,damp2):
    Rold = copy.copy(R)
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

    # 打印每次迭代的ΔR
    # R_deta = Rnew - Rold
    # print('R_deta ={}'.format(R_deta))
    R = damp2*Rnew + (1-damp2)*Rold
    return R




def update_Amatrix(G,A,R,damp2):
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
    # print("Anew = ")
    # print(Anew)
    # print("Aold = ")
    # print(Aold)

    #打印每次迭代的ΔA
    # A_deta = Anew - Aold
    # print('A_deta ={}'.format(A_deta))
    A = damp2 * Anew + (1 - damp2) * Aold
    return A

def sort_centers(G,centers,A,R):
    for i in range(G.number_of_nodes()):
        centers.append({'num_nodes':i,'value':A[i][i]+R[i][i]})
    # print(centers)
    centers.sort(key = lambda x:x['value'],reverse = True)
    return centers

def FRAP(G,n,N0,standard_reference,iter=10):
    S = np.zeros([G.number_of_nodes(), G.number_of_nodes()], dtype=float)  # 初始化相似度矩阵
    centers = list()  # 初始化一个frap收敛结果的列表
    damp1,damp2 = 0.9,0.9
    path = nx.all_pairs_shortest_path(G)  # 调用多源最短路径算法，计算图G节点间的最短路径
    # nx.draw_networkx(G)
    nodes_positions = {}
    nodes_positions = nx.drawing.spring_layout(G).copy()  # 节点位置词典，通过Fruchterman-Reingold算法排列节点
    # print(nodes_positions)
    for i in range(G.number_of_nodes()):  # 计算出相似度矩阵，欧氏距离及节点间距离加权和的赋值
        for j in range(G.number_of_nodes()):
            S[i][j] = -damp1 * (len(path[i][j]) - 1) + (1 - damp1) * dis(nodes_positions,i, j)
    s = 0.0
    for i in range(G.number_of_nodes()):  # 求出相似度矩阵的中位数作为对角线元素
        for j in range(G.number_of_nodes()):
            if (i != j):
                s += S[i][j]
    mean = s / (G.number_of_nodes() * (G.number_of_nodes() - 1))
    for i in range(G.number_of_nodes()):
        S[i][i] = mean
    # print('S =')
    # print(S)

    for i in range(G.number_of_nodes()):  # 将中位数赋值给S[u][u]
        S[i][i] = mean

    A = np.zeros([G.number_of_nodes(), G.number_of_nodes()], dtype=float)
    R = np.zeros([G.number_of_nodes(), G.number_of_nodes()], dtype=float)
    # iter = 10
    for i in range(iter):
        print("iter = %d" % (i + 1))
        # print()
        R1 = R
        R = update_Rmatrix(G,S,A,R,damp2)
        # print('R =')
        # print(R)
        A = update_Amatrix(G,A,R1,damp2)
        # print('A =')
        # print(A)
    # print(sort_centers(G,centers,A,R))
    A_R_seq = [x['num_nodes'] for x in sort_centers(G,centers,A,R)]
    # print(A_R_seq)
    # n,N0 = 4,1      #实验证明n =2，N0= 3的时候结果还行(dolphins)
    N = A_R_seq[:n]
    V_sub_N = A_R_seq[n:]
    reference = fast_unfolding(G,[])[-1]
    # 分配标签,用labels字典存储
    labels = dict((i, i) for i in G.nodes())
    break_flag = False
    for node1 in N:
        for node2 in N:
            if find_pos(node1,reference)==find_pos(node2,reference):
                labels[node2] = labels[node1]

    # print('将边缘孤立节点各自分配好后：{}'.format(labels))

    path = nx.all_pairs_shortest_path(G)
    # for i in range(N0):
    #     for node in A_R_seq[n:]:
    #         min = N[random.randint(0,len(N)-1)]
    #         for x in N:
    #             if len(path[node][x])<len(path[node][min]):
    #                 min = x
    if n!=0:
        for i in range(1, N0):
            for node in V_sub_N:
                length, isolate = len(path[node][N[0]]) - 1, N[0]
                for x in N:
                    if length > len(path[node][x]) - 1 and node in find_pos(x, reference):
                        length, isolate = len(path[node][x]) - 1, x
                if length <= N0:
                    labels[node] = labels[x]
                    V_sub_N.remove(node)
                    # print('shanchule:%d' % node)

    # print('将与边缘孤立节点路径长<=N0的节点 分配好后：{}'.format(labels))

    t, flag = 1, True
    while t <= 100 and flag == True:
        flag = False
        for x in A_R_seq:
            counts = np.bincount([labels[key] for key in G.neighbors(x)]).tolist()  # 统计当前节点邻居节点的各标签个数
            max_pos = np.argmax(counts)  # 记录最大标签数的索引
            same_list = []  # 存放可能的多个最大标签数
            for i, j in enumerate(counts):
                if j == counts[max_pos]:
                    same_list.append(i)
            if labels[x] not in same_list:
                labels[x] = random.sample(same_list, 1)[0]
                flag = True
        t += 1
    print('t={}'.format(t))
    result_coms1 = labels
    my_d = {}
    for key in result_coms1:
        if my_d.get(result_coms1[key]) == None:
            my_d[result_coms1[key]] = [key]
        else:
            my_d[result_coms1[key]].append(key)
    result_coms = list(my_d.values())
    # print('最后结果：{}'.format(labels))
    Q = calculate_Modulartiy(G, result_coms)
    NMI = calculate_NMI(result_coms, standard_reference)
    jaccard = calculate_jaccard(result_coms, standard_reference)
    return Q, NMI, jaccard, result_coms



def find_pos(x,coms):
    for com_c in coms:
        if x in com_c:
            return com_c



def simple_FRAP(G,n,N0,standard_reference,iter=10):
    S = np.zeros([G.number_of_nodes(), G.number_of_nodes()], dtype=float)  # 初始化相似度矩阵
    centers = list()  # 初始化一个frap收敛结果的列表
    damp1,damp2 = 0.9,0.9
    path = nx.all_pairs_shortest_path(G)  # 调用多源最短路径算法，计算图G节点间的最短路径
    # nx.draw_networkx(G)
    nodes_positions = {}
    nodes_positions = nx.drawing.spring_layout(G).copy()  # 节点位置词典，通过Fruchterman-Reingold算法排列节点
    # print(nodes_positions)
    for i in range(G.number_of_nodes()):  # 计算出相似度矩阵，欧氏距离及节点间距离加权和的赋值
        for j in range(G.number_of_nodes()):
            S[i][j] = -damp1 * (len(path[i][j]) - 1) + (1 - damp1) * dis(nodes_positions,i, j)
    s = 0.0
    for i in range(G.number_of_nodes()):  # 求出相似度矩阵的中位数作为对角线元素
        for j in range(G.number_of_nodes()):
            if (i != j):
                s += S[i][j]
    mean = s / (G.number_of_nodes() * (G.number_of_nodes() - 1))
    for i in range(G.number_of_nodes()):
        S[i][i] = mean
    # print('S =')
    # print(S)

    for i in range(G.number_of_nodes()):  # 将中位数赋值给S[u][u]
        S[i][i] = mean

    A = np.zeros([G.number_of_nodes(), G.number_of_nodes()], dtype=float)
    R = np.zeros([G.number_of_nodes(), G.number_of_nodes()], dtype=float)
    # iter = 10
    for i in range(iter):
        print("iter = %d" % (i + 1))
        # print()
        R1 = R
        R = update_Rmatrix(G,S,A,R,damp2)
        # print('R =')
        # print(R)
        A = update_Amatrix(G,A,R1,damp2)
        # print('A =')
        # print(A)
    # print(sort_centers(G,centers,A,R))

# print(G1.nodes())
# print(G2.nodes())
# print(G3.nodes())
# print('fast_unfolding = {}'.format(fast_unfolding(G3,[])))
# reference = fast_unfolding(G3,[])[-1]
# print('LPA = {}'.format(LPA(G3,reference)))
# print('LPAH = {}'.format(LPAH(G3,reference)))
# print('FRAP = {}'.format(FRAP(G3,3,1,reference,iter = 31)))

# standard_result_Q = calculate_Modulartiy(G4,Gtest_coms)
# # print('standard result = %.20f' %standard_result_Q)
# reference = Gtest_coms
# # print('fast_unfolding = {}'.format(fast_unfolding(G4,reference)))
# # print('LPA = {}'.format(LPA(G4,reference)))
# # print('LPAH = {}'.format(LPAH(G4,reference)))
# # print('FRAP = {}'.format(FRAP(G4,0,0,reference,iter = 0)))

#网格搜索选最优参数  并求出均值和标准差
# for n in range(10):
#     for N0 in range(10):
#         print('n = {}   N0 = {}'.format(n, N0))
#         li = []
#         for times in range(20):
#             li.append(FRAP(G1,n,N0)[0])
#         print('mean = {}   Std = {}    {}'.format(np.mean(li),np.std(li),li))
# print(nx.all_pairs_shortest_path(G2)[13][60])

# colors = cnames = {
# 'aliceblue':            '#F0F8FF',
# 'antiquewhite':         '#FAEBD7',
# 'aqua':                 '#00FFFF',
# 'aquamarine':           '#7FFFD4',
# 'azure':                '#F0FFFF',
# 'beige':                '#F5F5DC',
# 'bisque':               '#FFE4C4',
# 'black':                '#000000',
# 'blanchedalmond':       '#FFEBCD',
# 'blue':                 '#0000FF',
# 'blueviolet':           '#8A2BE2',
# 'brown':                '#A52A2A',
# 'burlywood':            '#DEB887',
# 'cadetblue':            '#5F9EA0',
# 'chartreuse':           '#7FFF00',
# 'chocolate':            '#D2691E',
# 'coral':                '#FF7F50',
# 'cornflowerblue':       '#6495ED',
# 'cornsilk':             '#FFF8DC',
# 'crimson':              '#DC143C',
# 'cyan':                 '#00FFFF',
# 'darkblue':             '#00008B',
# 'darkcyan':             '#008B8B',
# 'darkgoldenrod':        '#B8860B',
# 'darkgray':             '#A9A9A9',
# 'darkgreen':            '#006400',
# 'darkkhaki':            '#BDB76B',
# 'darkmagenta':          '#8B008B',
# 'darkolivegreen':       '#556B2F',
# 'darkorange':           '#FF8C00',
# 'darkorchid':           '#9932CC',
# 'darkred':              '#8B0000',
# 'darksalmon':           '#E9967A',
# 'darkseagreen':         '#8FBC8F',
# 'darkslateblue':        '#483D8B',
# 'darkslategray':        '#2F4F4F',
# 'darkturquoise':        '#00CED1',
# 'darkviolet':           '#9400D3',
# 'deeppink':             '#FF1493',
# 'deepskyblue':          '#00BFFF',
# 'dimgray':              '#696969',
# 'dodgerblue':           '#1E90FF',
# 'firebrick':            '#B22222',
# 'floralwhite':          '#FFFAF0',
# 'forestgreen':          '#228B22',
# 'fuchsia':              '#FF00FF',
# 'gainsboro':            '#DCDCDC',
# 'ghostwhite':           '#F8F8FF',
# 'gold':                 '#FFD700',
# 'goldenrod':            '#DAA520',
# 'gray':                 '#808080',
# 'green':                '#008000',
# 'greenyellow':          '#ADFF2F',
# 'honeydew':             '#F0FFF0',
# 'hotpink':              '#FF69B4',
# 'indianred':            '#CD5C5C',
# 'indigo':               '#4B0082',
# 'ivory':                '#FFFFF0',
# 'khaki':                '#F0E68C',
# 'lavender':             '#E6E6FA',
# 'lavenderblush':        '#FFF0F5',
# 'lawngreen':            '#7CFC00',
# 'lemonchiffon':         '#FFFACD',
# 'lightblue':            '#ADD8E6',
# 'lightcoral':           '#F08080',
# 'lightcyan':            '#E0FFFF',
# 'lightgoldenrodyellow': '#FAFAD2',
# 'lightgreen':           '#90EE90',
# 'lightgray':            '#D3D3D3',
# 'lightpink':            '#FFB6C1',
# 'lightsalmon':          '#FFA07A',
# 'lightseagreen':        '#20B2AA',
# 'lightskyblue':         '#87CEFA',
# 'lightslategray':       '#778899',
# 'lightsteelblue':       '#B0C4DE',
# 'lightyellow':          '#FFFFE0',
# 'lime':                 '#00FF00',
# 'limegreen':            '#32CD32',
# 'linen':                '#FAF0E6',
# 'magenta':              '#FF00FF',
# 'maroon':               '#800000',
# 'mediumaquamarine':     '#66CDAA',
# 'mediumblue':           '#0000CD',
# 'mediumorchid':         '#BA55D3',
# 'mediumpurple':         '#9370DB',
# 'mediumseagreen':       '#3CB371',
# 'mediumslateblue':      '#7B68EE',
# 'mediumspringgreen':    '#00FA9A',
# 'mediumturquoise':      '#48D1CC',
# 'mediumvioletred':      '#C71585',
# 'midnightblue':         '#191970',
# 'mintcream':            '#F5FFFA',
# 'mistyrose':            '#FFE4E1',
# 'moccasin':             '#FFE4B5',
# 'navajowhite':          '#FFDEAD',
# 'navy':                 '#000080',
# 'oldlace':              '#FDF5E6',
# 'olive':                '#808000',
# 'olivedrab':            '#6B8E23',
# 'orange':               '#FFA500',
# 'orangered':            '#FF4500',
# 'orchid':               '#DA70D6',
# 'palegoldenrod':        '#EEE8AA',
# 'palegreen':            '#98FB98',
# 'paleturquoise':        '#AFEEEE',
# 'palevioletred':        '#DB7093',
# 'papayawhip':           '#FFEFD5',
# 'peachpuff':            '#FFDAB9',
# 'peru':                 '#CD853F',
# 'pink':                 '#FFC0CB',
# 'plum':                 '#DDA0DD',
# 'powderblue':           '#B0E0E6',
# 'purple':               '#800080',
# 'red':                  '#FF0000',
# 'rosybrown':            '#BC8F8F',
# 'royalblue':            '#4169E1',
# 'saddlebrown':          '#8B4513',
# 'salmon':               '#FA8072',
# 'sandybrown':           '#FAA460',
# 'seagreen':             '#2E8B57',
# 'seashell':             '#FFF5EE',
# 'sienna':               '#A0522D',
# 'silver':               '#C0C0C0',
# 'skyblue':              '#87CEEB',
# 'slateblue':            '#6A5ACD',
# 'slategray':            '#708090',
# 'snow':                 '#FFFAFA',
# 'springgreen':          '#00FF7F',
# 'steelblue':            '#4682B4',
# 'tan':                  '#D2B48C',
# 'teal':                 '#008080',
# 'thistle':              '#D8BFD8',
# 'tomato':               '#FF6347',
# 'turquoise':            '#40E0D0',
# 'violet':               '#EE82EE',
# 'wheat':                '#F5DEB3',
# 'white':                '#FFFFFF',
# 'whitesmoke':           '#F5F5F5',
# 'yellow':               '#FFFF00',
# 'yellowgreen':          '#9ACD32'}

colors ={'peru':                 '#CD853F',
         'dodgerblue':           '#1E90FF',
         'orange': '#FFA500',
         'brown': '#A52A2A',
         'deeppink': '#FF1493',
         'deepskyblue': '#00BFFF',
         'greenyellow': '#ADFF2F',
         'darkred': '#8B0000',
         'darkolivegreen': '#556B2F',
         'purple': '#800080',
         'violet': '#EE82EE',
         'sienna': '#A0522D',
         'darkkhaki': '#BDB76B',
         'darksalmon': '#E9967A',
         'gold': '#FFD700',
         }

shapes = ['o','^','s','h','*']
# node_colors = [0]*len(G2.nodes())
# for x in fast_unfolding(G2)[1]:
#     color = colors.pop(random.sample(colors.keys(),1)[0])
#     for i in x:
#         node_colors[i] = color
# print(node_colors)
# nx.draw_networkx(G2,node_color = node_colors)
# plt.show()



def generate_circle_pos(center,r,n):
    positions = []
    for i in range(n):
        for k in range(20):
            x = [random.uniform(center[0] - r, center[0] + r), random.uniform(center[1] - r, center[1] + r)]
            vec1 = np.array(x)
            vec2 = np.array(center)
            print(vec1,vec2)
            if np.sqrt(np.sum(np.square(vec1 - vec2))) > r:
                continue
            else:
                positions.append(x)
                break

    return positions

# generate_circle_pos([0.25,0.25],0.15,10)


# # #使用不同颜色和形状画图,也就是用for循环来一次次分开list_node   (https://segmentfault.com/a/1190000000527216)
# partition = fast_unfolding(G2,[])[-1]
# pos = nx.spring_layout(G2)
# com_centers = [[0.25,0.25],[0.25,0.75],[0.5,0.5],[0.75,0.25],[0.75,0.75]]
# for com_c in partition :
#     # list_nodes = [nodes for nodes in partition
#     #                             if partition[nodes] == com]
#     # pos = generate_circle_pos(com_centers.pop(random.randint(0,len(com_centers)-1)),0.15,len(com_c))
#     nx.draw_networkx_nodes(G2, pos, com_c,labels = True, node_size = 90,
#                                 node_color = colors.pop(random.sample(colors.keys(),1)[0]),node_shape = shapes.pop(random.randint(0,len(shapes)-1)))
#
# nx.draw_networkx_edges(G2,pos,with_labels = True, alpha=0.5 )
# nx.draw_networkx_labels(G2,pos,font_size=8)
# plt.show()
# nx.draw_networkx(G2,pos)
# plt.show()

# partition = [[0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21], [24, 25, 28, 31], [4, 5, 6, 10, 16], [8, 9, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33]]
# pos = nx.spring_layout(G1)
# com_centers = [[0.25,0.25],[0.25,0.75],[0.5,0.5],[0.75,0.25],[0.75,0.75]]
# for com_c in partition :
#     # list_nodes = [nodes for nodes in partition
#     #                             if partition[nodes] == com]
#     # pos = generate_circle_pos(com_centers.pop(random.randint(0,len(com_centers)-1)),0.15,len(com_c))
#     nx.draw_networkx_nodes(G2, pos, com_c,labels = True, node_size = 150,
#                                 node_color = colors.pop(random.sample(colors.keys(),1)[0]),node_shape = shapes.pop(random.randint(0,len(shapes)-1)))
#
# nx.draw_networkx_edges(G1,pos,with_labels = True, alpha=0.5 )
# nx.draw_networkx_labels(G1,pos,font_size=8)
# plt.show()
# nx.draw_networkx(G1,pos)
# plt.show()



# partition = fast_unfolding(G2,[])[-1]
# pos = nx.spring_layout(G2)
# com_centers = [[0.25,0.25],[0.25,0.75],[0.5,0.5],[0.75,0.25],[0.75,0.75]]
# for com_c in partition :
#     # list_nodes = [nodes for nodes in partition
#     #                             if partition[nodes] == com]
#     # pos = generate_circle_pos(com_centers.pop(random.randint(0,len(com_centers)-1)),0.15,len(com_c))
#     nx.draw_networkx_nodes(G2, pos, com_c,labels = True, node_size = 90,
#                                 node_color = colors.pop(random.sample(colors.keys(),1)[0]),node_shape = shapes.pop(random.randint(0,len(shapes)-1)))
#
# nx.draw_networkx_edges(G2,pos,with_labels = True, alpha=0.5 )
# nx.draw_networkx_labels(G2,pos,font_size=8)
# plt.show()


# 将dolphin划分为五个区域画出图来！
partition = fast_unfolding(G2,[])[-1]
# pos = nx.spring_layout(G2)
com_centers = [[0.25,0.25],[0.25,0.75],[0.5,0.5],[0.75,0.25],[0.75,0.75]]
pos = {}
for com_c in partition :
    # list_nodes = [nodes for nodes in partition
    #                             if partition[nodes] == com]
    random_pos = generate_circle_pos(com_centers.pop(random.randint(0,len(com_centers)-1)),0.15,len(com_c))
    for x in com_c:
        pos[x] = random_pos.pop()
    nx.draw_networkx_nodes(G2, pos, com_c,labels = True, node_size = 90,
                                node_color = colors.pop(random.sample(colors.keys(),1)[0]),node_shape = shapes.pop(random.randint(0,len(shapes)-1)))

nx.draw_networkx_edges(G2,pos,with_labels = True, alpha=0.5 )
nx.draw_networkx_labels(G2,pos,font_size=8)
plt.show()
# nx.draw_networkx(G2,pos)
# plt.show()







# import random
# for i in range(20):
#     d = {1: 2, 3: 4, 5: 6, 7: 8}
#     d.pop(random.sample(d.keys(), 1)[0])
#     print(d)









'''      G.neighbors(node)函数用来求图G中node的邻居节点，nx.common_neighbors(G,node1,node2)用来求node1，node2的公共
          邻居节点(但返回的是一个generator，不是列表)    '''

# labels = dict((i,i) for i in G.nodes())
# print(sorted(G.neighbors(1)))
# print(np.bincount([labels[key] for key in G.neighbors(1)]))



# my_G.add_edges_from([(1,2),(1,3),(2,3)])
# nx.draw_networkx(my_G,node_color = ['r','g','b'],node_shape = ['o','^','s'])    #通过调参可以给不同节点赋予不同的颜色(node_color),(node_shape)不行
# plt.show()
# print(my_G.edges())

# # #统计非负数整数的众数
# a = np.array([1,2,3,1,2,1,1,1,3,2,2,1,3,3,3,3,5,5,5,5,5,5,100])
# counts = np.bincount(a).tolist()
# print(counts)
# max_pos = np.argmax(counts)
# same_list = []
# for i,x in enumerate(counts):
#     if x==counts[max_pos]:
#         same_list.append(i)
# print(same_list)
# print(random.sample(same_list,1))


# #一次lpa结果   （karate）
# d = {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 34, 11: 20, 12: 20, 13: 20, 14: 20, 15: 34, 16: 34, 17: 20, 18: 20, 19: 34, 20: 20, 21: 34, 22: 20, 23: 34, 24: 34, 25: 32, 26: 32, 27: 34, 28: 34, 29: 20, 30: 34, 31: 20, 32: 32, 33: 34, 34: 34}
# my_d = {}
# for key in d:
#     if my_d.get(d[key]) == None:
#         my_d[d[key]] = [key]
#     else:
#         my_d[d[key]].append(key)
# lpa_Q = calculate_Modulartiy(G,list(my_d.values()))
# print('lpa_Q= %f' % lpa_Q)

# my_G = nx.Graph()
# my_G.add_edges_from(G.edges())
# nx.draw(my_G)
# nx.draw_networkx(my_G)
# plt.show()
# print(my_G.edges())
# print(G.edges())
# print(my_G.nodes())
# print(G.nodes())

Gtest = nx.Graph()
Gtest_coms = {}

# with open('E:/社区 数据/LFR datasets/network.txt','r') as f:
#     for x in f:
#         Gtest.add_edge(int(x.strip().split()[0]),int(x.strip().split()[1]))
#
# with open('E:/社区 数据/LFR datasets/community.txt','r') as f:
#     for x in f:
#         node,community = int(x.strip().split()[0]),int(x.strip().split()[1])
#         if Gtest_coms.get(community)==None:
#             Gtest_coms[community] = [node]
#         else:
#             Gtest_coms[community].append(node)
#     Gtest_coms = list(Gtest_coms.values())
# print('standard result = %f' %(calculate_Modulartiy(Gtest,Gtest_coms)))
# print(fast_unfolding(Gtest))


# Gtest2 = nx.Graph()
# Gtest_coms2 = {}
#
# with open('E:/社区 数据/LFR datasets/network.dat','r') as f:
#     for x in f:
#         Gtest2.add_edge(int(x.strip().split()[0]),int(x.strip().split()[1]))
#
# with open('E:/社区 数据/LFR datasets/community.dat','r') as f:
#     for x in f:
#         node,community = int(x.strip().split()[0]),int(x.strip().split()[1])
#         if Gtest_coms2.get(community)==None:
#             Gtest_coms2[community] = [node]
#         else:
#             Gtest_coms2[community].append(node)
#     Gtest_coms2 = list(Gtest_coms2.values())
# print('standard result = %f' %(calculate_Modulartiy(Gtest2,Gtest_coms2)))
# print(fast_unfolding(Gtest2))



# #测试，可以通过手动设置node_position位置来指定pos参数！！
# G_fafdfafd = nx.Graph()
# G_fafdfafd.add_edges_from([(1,2),(2,3),(3,8)])
# G_fafdfafd.add_nodes_from([4,5,6,7])
# node_position = {1:[random.uniform(0,0.5),random.uniform(0,0.5)],2:[random.uniform(0,0.5),random.uniform(0,0.5)],
#                  3:[random.uniform(0.5,1),random.uniform(0,0.5)],4:[random.uniform(0.5,1),random.uniform(0,0.5)],
#                  5: [random.uniform(0,0.5), random.uniform(0.5,1)],6: [random.uniform(0, 0.5), random.uniform(0.5,1)],
#                  7: [random.uniform(0.5,1), random.uniform(0.5,1)],8: [random.uniform(0.5, 1), random.uniform(0.5,1)]}
# nx.draw_networkx(G_fafdfafd,node_position)
# plt.show()
# # for i in range(10):
# #     print(random.uniform(0,0.5))





# import matplotlib.pyplot as plt
# import pandas as pd
#
# f = open("E:/社区 数据/Results/my_result_pics.csv")
# mydata = pd.read_csv(f)
# print(mydata)
# x1,y1= mydata['x1'],mydata['y1']
# x2,y2= mydata['x2'],mydata['y2']
# x3,y3= mydata['x3'],mydata['y3']
#
# plt.plot(x1,y1,color ='green',label = 'Affinity propagation',linestyle ='-')
# plt.plot(x2,y2,color ='red',label = 'k-means',linestyle ='--')
# plt.plot(x3,y3,color ='blue',label = 'Random',linestyle ='-.')
#
# plt.xlim((0,5.0))
# plt.ylim((0,50))
# plt.xlabel('False positive rate(%)')
# plt.ylabel('True positive rate(%)')
# plt.legend(loc = 'upper left')
# plt.show()


