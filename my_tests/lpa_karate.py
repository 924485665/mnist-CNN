import networkx as nx
import matplotlib.pyplot as plt
import numpy
from numpy import *
import time
import copy

def lpa(current_belongs):

    for i in range(G.number_of_nodes()):
        neighbors = list()  # 存储第 i+1个节点的邻居结点
        labels = list()  # labels 统计这些邻居结点的各自社区次数
        neighbors = get_neighbors(i)
        # if(len(neighbors) != 0):
        #     for nodes in range(len(neighbors)):
        #         labels.append({'community': neighbors[nodes], 'times': 0})
        #     for nodes in range(len(neighbors)):
        #         for i in range(len(labels)):
        #             if (labels[i]['community'] == neighbors[nodes]):
        #                 labels[i]['times'] += 1
        #     print(labels)
        #     labels.sort(key=lambda x: x['times'], reverse=True)
        #     print(labels)
        #     belongs[i]['community'] = labels[0]['community']
        if (len(neighbors) != 0):
            for nodes in range(len(neighbors)):
                flag = False
                for i in range(len(labels)):
                    if(current_belongs[neighbors[nodes]]['community'] == labels[i]['community']):
                        labels[i]['times'] += 1
                        flag = True
                        break
                if(flag == False):
                    labels.append({'community': current_belongs[nodes], 'times': 1})


            print(labels)
            labels.sort(key=lambda x: x['times'], reverse=True)
            print(labels)
            current_belongs[i]['community'] = labels[0]['community']

    return current_belongs










def get_neighbors(i):                               #获得第(i+1)个结点的邻居节点列表并返回
    neis = list()
    for i in range(G.number_of_edges()):
        for k in range(G.number_of_edges()):
            if((i+1) == G.edges()[k][0] or (i+1) == G.edges()[k][1]):
                if((i+1) == G.edges()[k][0]):
                    neis.append(G.edges()[k][1])
                else:
                    neis.append(G.edges()[k][0])


    return neis









G = nx.read_gml('E:\社区 数据\karate\karate.gml')
belongs = list()
for i in range(G.number_of_nodes()):
    belongs.append({'nodes':i+1,'community':i+1})

print(belongs)
belongs = lpa(belongs)
print(belongs)
