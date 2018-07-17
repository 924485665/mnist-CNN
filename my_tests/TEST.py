# coding = utf-8
#  d={'name':'Tom','age':'22'}
# print(d.get('name'))
# print (d.get('test'))
#
# print (d.setdefault('name'))
# print (d.setdefault('test_1'))
# print (d.setdefault('test_2',80))
# print (d)
#
# def get_seq(n):
#     s = []
#     i = 2
#     while True:
#         j = 2
#         while j <= i ** 0.5:
#             if (i % j) == 0:
#                 break
#             j += 1
#         if j == int(i ** 0.5) + 1:
#             s.append(i)
#         if len(s) == n:
#             break
#         i +=1
#
#
#     return s
#
#
# if __name__ == '__main__':
#     a = int(input())
#     s = []
#     max = 0
#     for i in range(a):
#         x = int(input())
#         s.append(x)
#         if x>max:
#             max = x
#     seq = get_seq(max)
#     for i in range(a):
#         print(seq[s[i]-1])




# def get_weight(s):
#     sum = 0
#     weight = 0
#     for i in range(len(s)):
#         if weight + s[i]< sum//2:
#             weight += s[i]
#         if -1<=weight-sum//2<=1:
#             return weight
#
#
# if __name__ == '__main__':
#     a = input().split()
#     n = int(a[0])
#     s = []
#     for i in range(1, n):
#         s.append(int(a[i]))
#     print(n, s)


# d = dict((i,10) for i in range(3))
# print(d)
# d[0] = 10
# print(d)
if 0 != None:
    print(True)
# str = raw_input()
# print(type(str))
# print(str.split())
# print 13/34.0
# import numpy
# f = [17,13,13,13,11,5,5,5]
# g = [0.38,1,1,1,0,0,0,0]
# sigema1 = numpy.sqrt(sum([x**2 for x in f]))
# sigema2 = numpy.sqrt(sum([x**2 for x in g]))
# print sigema1,sigema2
# p = []
# for i in range(len(f)):
#     p.append(f[i]/float(sigema1)+g[i]/float(sigema2))
# print p
#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# f = open('D:/qq文件/924485665/FileRecv/MobileFile/可视化/history150.csv')
# data = pd.read_csv(f)
# print(data)
# data['loss'].plot()
# data['val_loss'].plot()
# plt.title('my_test')
# plt.legend(loc = 'best')
# plt.show()


# f = open('D:/titanic/ROC_test.csv')
# # f = open("D:/titanic/my_result_pics.csv",'r',encoding='utf-8')
# mydata = pd.read_csv(f)
# print(mydata)
# x1,y11= mydata['x1'],mydata['y1']
# x2,y22= mydata['x2'],mydata['y2']
# x3,y33= mydata['x3'],mydata['y3']
#
# plt.plot(x1,y11,marker='^',color ='green',label = 'LPA',linestyle ='-',)
# plt.plot(x2,y22,marker='s',color ='red',label = 'LPAH',linestyle ='--')
# plt.plot(x3,y33,marker='o',color ='blue',label = 'FRAP',linestyle ='-.')
#
# plt.xlim((0,5.4))
# plt.ylim((0,45))
# plt.xticks(np.arange(0,5.2,1))
# plt.yticks(np.arange(0,45,10))
# plt.xlabel('False positive rate(%)')
# plt.ylabel('True positive rate(%)')
# plt.legend(loc = 'upper left')
# plt.show()

print([0.07/0.34,0.03/0.49,0.03/0.57])
print(np.mean([0.07/0.34,0.03/0.49,0.03/0.57]))
print(np.mean([0.39/0.425,0.05/0.79,0.04/0.915]))
print(np.mean([0.34/0.386,0.04/0.686,0.24/0.538]))


# import numpy as np
# li = list(range(20))
# print(li)
# li_np = np.array(li)
# print(li_np)
# print(np.mean(li))
# print(np.std(li))
#
# li_np2 = np.array([[1,0,0],[4,5,6]])
# print(li_np2)
# print(type(li_np2))
# print([li_np2[i][j] if li_np2[i][j]!=0 else 100  for i in range(len(li_np2)) for j in range(len(li_np2[0]))])
# s1 = {(1,2),(3,4),(5,6)}
# s2 = {(4,3),(3,4),(5,6)}
# print(s1&s2)
# print(s1|s2)

# import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np

# f = open('E:/社区 数据/Results/my_result_pics.csv')
# # f = open("D:/titanic/my_result_pics.csv",'r',encoding='utf-8')
# mydata = pd.read_csv(f)
# print(mydata)
# x1,y11= mydata['x1'],mydata['y11']
# x2,y22= mydata['x2'],mydata['y22']
# x3,y33= mydata['x3'],mydata['y33']
#
# plt.plot(x1,y11,marker='^',color ='green',label = 'LPA',linestyle ='-',)
# plt.plot(x2,y22,marker='s',color ='red',label = 'LPAH',linestyle ='--')
# plt.plot(x3,y33,marker='o',color ='blue',label = 'FRAP',linestyle ='-.')
#
# plt.xlim((0.1,0.51))
# plt.ylim((0,0.71))
# plt.xticks(np.arange(0.1,0.51,0.1))
# plt.yticks(np.arange(0,0.7,0.2))
# plt.xlabel('mu')
# plt.ylabel('Q')
# plt.legend(loc = 'best')
# plt.show()



# #计算标准差
# matrix_karate = np.array([[0.3293,0.3737,0.3419,0.3673],[0.3549,0.3520,0.3717,0.3194],[0.4151,0.4020,0.3944,0.4151]])
# print([np.std(x) for x in matrix_karate])
#
# matrix_dolphins = np.array([[0.4985,0.4985,0.4972],[0.5032,0.5032,0.4716],[0.5210,0.5210,0.5210]])
# print([np.std(x) for x in matrix_dolphins])
#
# matrix_football = np.array([[0.575,0.576,0.593,0.569,0.578,0.601],[0.533,0.581,0.582,0.571,0.581,0.581,0.603],[0.589,0.591,0.601,0.580,0.595,0.602,0.605]])
# print([np.std(x) for x in matrix_football])