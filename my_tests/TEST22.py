import networkx as nx
import community as com


def calculate_Modulartiy(com_c):

    m = len(G.edges())                         #m 为图中的总边数
    lc = 0                                     #lc为社区c内部边的条数
    Oc = 0
    Dc = 0
    for each_edge in range(m):
        if(G.edges()[each_edge][0] in com_c  and G.edges()[each_edge][1] in com_c ):
            lc +=1
        else:
            if((G.edges()[each_edge][0] in com_c  and G.edges()[each_edge][1] not in com_c)or (G.edges()[each_edge][0] not in com_c  and G.edges()[each_edge][1] in com_c)):
                Oc +=1
    Dc = 2*lc + Oc

    return(lc/m - (Dc/(2*m))*(Dc/(2*m)))

G = nx.read_gml('E:\社区 数据\karate\karate.gml')
partition = com.best_partition(G)
print(partition)
print(partition.values())
print(partition.keys())
com_dict = {0:[],1:[],2:[],3:[]}

for i in partition.keys():
    if partition[i] == 0:
        com_dict[0].append(i)
    elif partition[i] == 1:
        com_dict[1].append(i)
    elif partition[i] == 2:
        com_dict[2].append(i)
    else:
        com_dict[3].append(i)

print(com_dict)
Q = calculate_Modulartiy(com_dict[0])+calculate_Modulartiy(com_dict[1])+calculate_Modulartiy(com_dict[2])+calculate_Modulartiy(com_dict[3])
print('Q = %f' %Q)