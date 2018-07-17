from igraph import *
import numpy
from numpy import *
import time
import copy
# g = Graph.Read_GML("E:\社区 数据\karate\karate.gml")
# g.write_pajek()
labels = list()
for i in range(10):
    labels.append({'com':i+1,'times':i})


print(labels)
x = max(y['com'] for y in labels)
print(x)
f =lambda x,y:x+y
labels[f(1,1)]['times'] = 10
print(labels)
print([x['com'] for x in labels])

l = list()
print(len(l))
break_flag=False
for i in range(10):
    print("爷爷层")
    for j in range(10):
        print("爸爸层")
        for k in range(10):
            print("孙子层")
            if k==3:
                break_flag=True
                break                    #跳出孙子层循环，继续向下运行
        print("haha!!")
        if break_flag==True:
            break                        #满足条件，运行break跳出爸爸层循环，向下运行
    print("hahaha!!")
    if break_flag==True:
        break                            #满足条件，运行break跳出爷爷层循环，结束全部循环，向下运行
print("keep going...")

a = list()
for i in range(3):
    a.append({'nodes': i + 1, 'community': i + 1})

def lpa(list):
    print("in the function  %s" %list)
    list[2]['community'] = 10
    print("in the function  %s" %list)
    return list



print(a)
a =lpa(a)
print(a)