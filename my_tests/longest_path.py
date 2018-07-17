#coding:utf-8
# def get_neis(v):
#     neis = []
#     if not d.get(v):
#         return []
#     else:
#         for i in [int(x[1]) for x in d[v]]:
#             if not path[i]:
#                 neis.append(i)
#         return neis
#
# def get_neis2(v):
#     neis = []
#     if not d.get(v):
#         return []
#     else:
#         for i in [int(x[1]) for x in d[v]]:
#             if not book[i]:
#                 neis.append(i)
#         return neis
#
# def get_neis_all(v):
#     neis = []
#     if not d.get(v):
#         return []
#     else:
#         return [int(x[1]) for x in d.get(v)]
#
#
# def get_dis(i,j):
#     if not d.get(i):
#         return False
#     if j not in [int(x[1]) for x in d[i]]:
#         return False
#     for x in d[i]:
#         if int(x[1])==j:
#             return int(x[0])
#
# def DFS(cur,dis):
#     if cur == end and dis != 0:    #判断是否到达了目标点
#         for i in range(n):
#             if path[i]:
#                 print(i,end = ' ')
#         print(end)
#         print('路径长度为  %d' %dis)
#
#         if min[0] >dis :
#             min[0] = dis
#         return
#
#     for i in range(n):
#         current_dis = get_dis(cur,i)
#         if current_dis and not path[i]:
#             path[i] = True
#             DFS(i,dis + current_dis)
#             path[i] = False
#
# def DFS2(cur,dis):
#     if cur==end and dis:
#         for i in range(n):
#             if path[i]:
#                 print('{}-'.format(i),end = ' ')
#         print('{}'.format(end))
#     neis = get_neis_all(cur)
#     if neis:
#         for i in neis:
#             if not path[i]:
#                 path[i] = True
#                 DFS2(i,dis+get_dis(cur,i))
#                 path[i] = False
#     return

# #深度优先搜索图的路径
# def DFS2(cur,dis):
#     path[cur] = True
#     neis = get_neis(cur)
#     if len(neis) == 0 and get_dis(cur,0):    #判断是否到达了目标点
#         for i in range(n):
#             if path[i]:
#                 print(i,end = ' ')
#         print(0,end = ' ')
#         dis += get_dis(cur,0)
#         print('路径长度为  %d' %(dis))
#
#         if min[0] >dis :
#             min[0] = dis
#         # path[cur] = False                     #return 之前把cur标记为未访问，不影响其他路径的深度搜索
#         return
#
#     for i in neis:
#         current_dis = get_dis(cur,i)
#         if current_dis and not path[i]:
#             path[i] =True
#             DFS(i,dis + current_dis)
#             path[i] = False
#     path[cur] = False                         #return 之前把cur标记为未访问，不影响其他路径的深度搜索



# stack = []
# book = [False] *5
# def dfs():   #非递归
#     stack.append(0)
#     book[0] = True
#     while len(stack):
#         v = stack[-1]
#         while len(get_neis2(v)):
#             v = get_neis2(v)[0]
#             stack.append(v)
#         if get_dis(v,0):
#             print(stack)
#         k = stack.pop()
#         book[k] = True
#         for i in get_neis_all(k):
#             if i and i != stack[-1]:
#                 book[i] = False


# def get_edge(G):
#     v0,v1,w = [],[],[]
#     for i in G:
#         for j in G[i]:
#             if G[i][j] :
#                 v0.append(i)
#                 v1.append(j)
#                 w.append(G[i][j])
#     return v0,v1,w
#
#
#
# def Bellman_Ford2(G,start,INF = 999):
#     v0,v1,w = get_edge(G)
#     dis = dict((key,INF) for key in G)
#     dis[start] = 0
#     for k in range(len(G)-1):              #进行 n-1次 松弛检验
#         flag = False
#         for i in range(len(w)):                  #对每条边都进行松弛检验
#             if dis[v1[i]]>dis[v0[i]] + w[i]:
#                 dis[v1[i]] = dis[v0[i]] + w[i]
#                 flag = True
#         if flag == False:break
#
#     #检查是否有负权回路
#     for i in range(len(w)):                   #尝试对每条边进行一次松弛检验
#         if dis[v1[i]] > dis[v0[i]] + w[i]:
#             return False
#     return dis

# if __name__ == '__main__':
#     n = 5
#     path = [False] * n
#     start,end = 0,0
#     min = [float('inf')]
#     m = int(input())
#     L = []
#     d = {}
#     for i in range(m):
#         L.append(input())
#     # for i in range(m):         #用字典d 来存储图，形式如：{0: ['21', '54', '12'], 1: ['44', '33'], 3: ['60'], 4: ['30']}
#     #     for j in range(0,len(L[i])-2,2):
#     #         if d.get(int(L[i][j]))==None:
#     #             d[int(L[i][j])]=[]
#     #         if L[i][j+1:j+3] not in d[int(L[i][j])]:
#     #             d[int(L[i][j])].append(L[i][j+1:j+3])
#
#     for x in L:         #用字典d 来存储图，形式如：{0: {1:2, 4:5, 2:1}, 1: {4:4, 3:3}, 3: {0:6}, 4: {0:3}}
#         for i in range(0,len(x)-2,2):
#             if not d.get(x[i]):
#                 d[x[i]] = {}
#             d[x[i]][x[i+2]] = int(x[i+1])
#             # print(L)
#     print(d)
#     Bellman_Ford2(d,'A')

    # print(d)
    # # for i in range(5):
    # #     print(get_neis(i))
    # DFS2(0,0)
    # # dfs()
    # print(min)








# Dijkstra算法——通过边实现松弛
# 指定一个点到其他各顶点的路径——单源最短路径

# 初始化图参数
# INF=999
# G = {1:{1:0,    2:1,    3:12},
#      2:{2:0,    3:9,    4:3,},
#      3:{3:0, 5:5},
#      4:{3:4,    4:0,    5:13,   6:15},
#      5:{5:0,    6:4},
#      6:{6:0}}
#
#
# # 每次找到离源点最近的一个顶点，然后以该顶点为重心进行扩展
# # 最终的到源点到其余所有点的最短路径
# # 一种贪婪算法
#
# def Dijkstra(G,v0):
#     """ 使用 Dijkstra 算法计算指定点 v0 到图 G 中任意点的最短路径的距离
#         INF 为设定的无限远距离值
#         此方法不能解决负权值边的图
#     """
#     book = set()
#     minv = v0
#
#     # 源顶点到其余各顶点的初始路程
#     dis = dict((k,INF) for k in G.keys())
#     dis[v0] = 0
#     print('dis={}\nG={}'.format(dis,G))
#
#
#     while len(book)<len(G):
#         book.add(minv)                                  # 确定当期顶点的距离
#         for w in G[minv]:                               # 以当前点的中心向外扩散
#             if dis[minv] + G[minv][w] < dis[w]:         # 如果从当前点扩展到某一点的距离小与已知最短距离
#                 dis[w] = dis[minv] + G[minv][w]         # 对已知距离进行更新
#
#         new_shortest = INF  # 从剩下的未确定点中选择最小距离点作为新的扩散点
#         for v in dis.keys():
#             if v in book: continue
#             if dis[v] < new_shortest:
#                 new_shortest = dis[v]
#                 minv = v
#     return dis
#
#
# dis = Dijkstra(G,v0=1)
# print(dis)




# # 初始化图参数
# def Dijkstra2(G,v0,INF=999):
#     dis = dict((key,INF) for key in G)
#     dis[v0]=0
#     book = set()
#     minv = v0
#     while len(book)<len(G):
#         book.add(minv)
#         for i in G[minv]:
#             if dis[i]>dis[minv]+G[minv][i]:
#                 dis[i] = dis[minv]+G[minv][i]
#         new_shortest = INF
#         for i in G:
#             if i in book:continue
#             if new_shortest>dis[i]:
#                 new_shortest = dis[i]
#                 minv = i
#     return dis
#
#
# def Dijkstra3(G,v0):     #自制错误版
#     dis = dict((key,INF) for key in G)
#     dis[v0] = 0
#     book= list()
#     minv = v0
#     while len(book)<len(G):
#         book.append(minv)
#         new_shortest = INF
#         next = minv
#         for i in G[minv]:
#             if dis[i]>dis[minv] +G[minv][i]:
#                 dis[i] = dis[minv]+G[minv][i]
#             if i in book:continue
#             if dis[i]<new_shortest:
#                 new_shortest = dis[i]
#                 next = i                      #错误在不见得每次新加入的节点一定是当前minv的邻居节点
#         minv = next
#     return dis
#
#
# if __name__ == '__main__':
#     result = Dijkstra2(G,1)
#     print(result)




# Bellman-Ford核心算法
# 对于一个包含n个顶点，m条边的图, 计算源点到任意点的最短距离
# 循环n-1轮，每轮对m条边进行一次松弛操作

# 定理：
# 在一个含有n个顶点的图中，任意两点之间的最短路径最多包含n-1条边
# 最短路径肯定是一个不包含回路的简单路径（回路包括正权回路与负权回路）
# 1. 如果最短路径中包含正权回路，则去掉这个回路，一定可以得到更短的路径
# 2. 如果最短路径中包含负权回路，则每多走一次这个回路，路径更短，则不存在最短路径
# 因此最短路径肯定是一个不包含回路的简单路径，即最多包含n-1条边，所以进行n-1次松弛即可


G2 = {1:{1:0, 2:-3, 5:5},
     2:{2:0, 3:2},
     3:{3:0, 4:3},
     4:{4:0, 5:2},
     5:{5:0}}



# def getEdges(G):
#     """ 输入图G，返回其边与端点的列表 """
#     v1 = []     # 出发点
#     v2 = []     # 对应的相邻到达点
#     w  = []     # 顶点v1到顶点v2的边的权值
#     for i in G:
#         for j in G[i]:
#             if G[i][j] != 0:
#                 w.append(G[i][j])
#                 v1.append(i)
#                 v2.append(j)
#     return v1,v2,w
#
# class CycleError(Exception):
#     pass
#
# def Bellman_Ford(G, v0, INF=999):
#     v1,v2,w = getEdges(G)
#
#     # 初始化源点与所有点之间的最短距离
#     dis = dict((k,INF) for k in G.keys())
#     dis[v0] = 0
#
#     # 核心算法
#     for k in range(len(G)-1):   # 循环 n-1轮
#         flag = False           # 用于标记本轮松弛中dis是否发生更新
#         for i in range(len(w)):     # 对每条边进行一次松弛操作
#             if dis[v1[i]] + w[i] < dis[v2[i]]:
#                 dis[v2[i]] = dis[v1[i]] + w[i]
#                 flag = True
#         if not flag: break
#
#     # 检测负权回路
#     # 如果在 n-1 次松弛之后，最短路径依然发生变化，则该图必然存在负权回路
#     flag = 0
#     for i in range(len(w)):             # 对每条边再尝试进行一次松弛操作
#         if dis[v1[i]] + w[i] < dis[v2[i]]:
#             flag = 1
#             break
#     if flag == 1:
# #         raise CycleError()
#         return False
#     return dis

# def get_edge(G):
#     v0,v1,w = [],[],[]
#     for i in G:
#         for j in G[i]:
#             if G[i][j] :
#                 v0.append(i)
#                 v1.append(j)
#                 w.append(G[i][j])
#     return v0,v1,w
#
#
#
# def Bellman_Ford2(G,start,INF = 999):
#     v0,v1,w = get_edge(G)
#     dis = dict((key,INF) for key in G)
#     dis[start] = 0
#     for k in range(len(G)-1):              #进行 n-1次 松弛检验
#         flag = False
#         for i in range(len(w)):                  #对每条边都进行松弛检验
#             if dis[v1[i]]>dis[v0[i]] + w[i]:
#                 dis[v1[i]] = dis[v0[i]] + w[i]
#                 flag = True
#         if flag == False:break
#
#     #检查是否有负权回路
#     for i in range(len(w)):                   #尝试对每条边进行一次松弛检验
#         if dis[v1[i]] > dis[v0[i]] + w[i]:
#             return False
#     return dis
#
# def Floyd_singlesource(G,v0):
#     for k in range(len(G)):
#         for i in range(len(G)):
#             if G[v0][i]>G[i][k]+G[k][i]:
#                 G[v0][i] = G[i][k] + G[k][i]
#     return G[v0]
# # dis = Bellman_Ford(G2, 1)
# # print(dis.values())
# dis2 = Bellman_Ford2(G2, 1)
# print(list(dis2.values()))

# E = [[0 for i in range(len(G2))] for j in range(len(G2))]
# for i in range(len(E)):
#     for j in range(len(E)):
#         if G2[i+1].get(j+1)== None:
#             E[i][j] = float('inf')
#         else:
#             E[i][j] = G2[i+1].get(j+1)
# dis3 = Floyd_singlesource(E, 1)
# print(dis3)

# # floyd 算法
# def Floyd(n):
#     for k in range(n):
#         for i in range(n):
#             if E[i][k] == float('inf'):
#                 continue
#             for j in range(n):
#                 if E[i][j]>E[i][k]+E[k][j]:
#                     E[i][j] = E[i][k]+E[k][j]

# if __name__ == '__main__':
#     E = [[0,2,6,4],[float('inf'),0,3,float('inf')],[7,float('inf'),0,1],[5,float('inf'),12,0]]
#     print(E)
#     Floyd(len(E))
#     print(E)





#网格搜索 从最上方到右下方的总共路径数（4行3列）

# def DFS(x,y):   #递归
#     if (x,y) ==(3,2):
#         print('(0,0)',end = '')
#         for i in stack:
#             print('--{}'.format(i),end = '')
#         print()
#         count[0] += 1
#
#     for pos in [(x,y+1),(x+1,y)]:
#         if pos[0]<4 and pos[1]<3:
#             stack.append(pos)
#             DFS(pos[0],pos[1])
#             stack.pop()


# book = [[False for i in range(3)] for j in range(4)]   #路径记录数组
# def dfs():   #非递归
#     stack.append((0,0))
#     book[0][0] = True
#     while stack:
#         v = stack[-1]
#         while len(get_myneis(v)):
#             v = get_myneis(v)[0]
#             stack.append(v)
#         if v == (3,2):
#             print(stack)
#             count[0]+=1
#         k = stack.pop()
#         book[k[0]][k[1]] = True
#         neis = get_all_neis(k)
#         for pos in neis:
#             book[pos[0]][pos[1]] = False
#
#
# def dp_nums():
#     dp = [[0 for i in range(3)] for j in range(4)]
#     dp[0][0] = 1
#     for i in range(4):
#         for j in range(3):
#             for x in [(i-1,j),(i,j-1)]:
#                 if x[0]>=0 and x[1]>=0:
#                     dp[i][j] +=dp[x[0]][x[1]]
#
#
#     print(dp)
#
#
#
# def get_myneis(v):
#     neis = []
#     for pos in [(v[0],v[1]+1),(v[0]+1,v[1])]:
#         if pos[0]<4 and pos[1]<3 and not book[pos[0]][pos[1]]:
#             neis.append(pos)
#
#     return neis
#
# def get_all_neis(k):
#     neis = []
#     for pos in [(k[0], k[1] + 1), (k[0] + 1, k[1])]:
#         if pos[0] < 4 and pos[1] < 3:
#             neis.append(pos)
#
#     return neis




# if __name__ == '__main__':
#     stack,count = [],[0]
#     # DFS(0,0)
#     dfs()
#     print(count[0])
#     dp_nums()


# #最长回文字串
# def longestPalindrome(s):
#     dp = [[False for i in range(len(s))] for j in range(len(s))]
#     max_length,start = 1,0
#     for i in range(len(s)):
#         for j in range(i+1):
#             if i-j<2:
#                 dp[j][i] = s[j]==s[i]
#             else:
#                 dp[j][i] = s[j]==s[i] and dp[j+1][i-1]
#                 if max_length<i-j+1 and dp[j][i]:
#                     max_length = i-j+1
#                     start = j
#
#     print(dp)
#     return s[start:start+max_length]
# print(longestPalindrome('abbaabcdacadcba'))




# 5
# A2B4E
# A5E
# A2B3D
# A1C
# D6A
# 5
# 02144
# 05430
# 02133
# 012
# 360

# 4
# 4 2
# 0 1 （从结点0到结点1的一条有向边）
# 1 2 （从结点1到结点2的一条有向边）
# 2 3 （从结点2到结点3的一条有向边）
# 0 2 （从结点0到结点2的一条有向边）


# 6 2
# a
# bc
# d
# eba
# ebc
# f
#
# ebcc
# ebd



# // line com
# /*fdasfdasfasd*/
# int
#    fdsafa
#    fdaf
# }
# s = 'fdafadfs"adf'
# x = s.find('"')
# s = s[:x]+s[x+1:]
# print(s)

# if __name__ == '__main__':
#     W = []
#     stack = []
#     sum = 0
#     for i in range(len(W)):
#         sum += W[i]
#     result = sum
#     results = []
#     T = 0
#     k = 0
#     line = raw_input().strip().split()
#     for x in range(1,len(line)):
#         W.append(int(line[x]))
#         sum +=int(line[x])
#     # print(W)
#     # print(sum)
#     diff = sum
#     while stack or k < len(W):
#         if len(stack) == 0:
#             stack.append(k)
#             T += W[k]
#             k += 1
#
#         while k < len(W):
#             if len(stack) + 1<= len(W) // 2 + len(W)%2:
#                 stack.append(k)
#                 T += W[k]
#             k += 1
#
#             if abs(sum - 2 * T) < diff :
#                 result = T
#                 diff = abs(sum - 2 * T)
#                 results = list(stack)
#
#         k = stack.pop()
#         T -= W[k]
#         k += 1
#
#     print min(result, sum - result), max(result, sum - result)
#     # print(results)

#阿里 光明小学 图题
# if __name__=='__main__':
#     n = int(raw_input())
#     m = int(raw_input())
#     x = raw_input()
#     map = []
#     for i in range(n):
#         line = [int(x) for x in raw_input().split()]
#         map.append(list(line))
#     dp = [list(x) for x in map]     #一维列表浅拷贝用list(),二维列表浅拷贝用[list(x) for x in dp]  三维用[[list(i) for i in x] for x in dp]
#     last_dp = [list(x) for x in map]
#     for k in range(m - 1):
#         for i in range(n):
#             for j in range(n):
#                 tmp = [last_dp[i][x] + map[x][j] for x in range(n) if x != i and x != j]
#                 dp[i][j] = min(tmp)
#         # copy
#         last_dp = [list(x) for x in dp]
#
#     print dp




# if __name__=='__main__':
#     n = int(raw_input())
#     m = int(raw_input())
#     x = raw_input()
#     map = []
#     for i in range(n):
#         line = [int(x) for x in raw_input().split()]
#         map.append(list(line))
#     print n,m
#     print map
#
#     last_dp = [list(x) for x in map]
#     print [id(x) for x in map]
#     print [id(x) for x in last_dp]
#     map[2]  = 100000
#     print map
#     print last_dp


# import numpy as np
# dp = [[[0]*3 for i in range(4)]for j in range(5)]
# print dp
# np_dp = np.array(dp)
# print np_dp.shape
# print np_dp
# dp2 = [[list(i) for i in x] for x in dp]
# dp[0][0][0] = 10
# print dp2
# print dp