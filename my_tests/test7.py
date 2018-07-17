import numpy as np
import pandas as pd
# e = np.zeros([4,4],dtype = float)
# print(e)
# e =[[0,2,1,4],[2,0,4,1],[1,4,0,100000],[4,1,100000,0]]
# for k in range(4):
#     for i in range(4):
#         for j in range(4):
#             if e[i][j]>e[i][k]+ e[k][j]:
#                 e[i][j] = e[i][k] + e[k][j]
#
# print(e)
# a = np.array([[0,2,1,4],[2,0,4,1],[1,4,0,100000],[4,1,100000,0]])
# e =[[0,2,1,4],[2,0,4,1],[1,4,0,100000],[4,1,100000,0]]
# print('a='+str(type(a)))
# print(a)
# print(a.shape)
# print(a.reshape(2,8))
# print(a.reshape((2,8),order=1))
# b= np.array([1.1,2,3,4])
# print(type(True))

# d = {'1':5}
# d['3']=10
# print(d)
# print(d.get('2'))
# print('2' in d)
# l = 'A2B4E'
# print(l[0:5])
# d['4']=[]
# d['4']=l[:5]
# print(d)
# print(len(d))
#
# 5
# A2B4E
# A5E
# A2B3D
# A1C
# D6A
# a = b = c = 0
# print('a = {}  b = {}  c = {}'.format(a,b,c))
#
# class A:
#     k = 10
#     def __init__(self,b):
#         self.a = b
#     def pri(self):
#         print('类属性 = {}  实例属性 = {}'.format(self.k,A.a))
#
# a1 = A(1000)
# a1.pri()
df = pd.DataFrame({'key1':list('aabba'),
                  'key2': ['one','two','one','two','one'],
                  'data1': np.random.randn(5),
                  'data2': np.random.randn(5)})
print(df)
print(df['data1'].apply(np.sqrt))