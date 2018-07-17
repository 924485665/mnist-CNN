#-*- coding:utf-8 -*-

# class Solution:
#     def judgeCircle(self, moves):
#         """
#         :type moves: str
#         :rtype: bool
#         """
#         return moves.count('L')==moves.count('R') and moves.count('U')==moves.count('D')
#
# s = input()
# print(s)
# s = s.replace('\"','')
# s = s.replace('\'','')
# print(s)
# obj = Solution()
# print(obj.judgeCircle(s))

# n = int(input())
# l = list()
# for i in range(n):
#     l.append(int(input()))
# for i in range(n):
#     if l[i]%2 != 0:
#         print('No')
#         continue
#     flag = False
#     for y in range(2,l[i]//2,2):
#         if (l[i]//y)%2 == 1:
#             print('%d %d' %(l[i]//y,y))
#             flag = True
#             break
#     if flag == False:
#         print('No')


#
# n = int(input())
# l = list()
# for i in range(n):
#     l.append(int(input()))
# for i in range(n):
#     if l[i]%2 != 0:
#         print('No')
#         continue
#     y = l[i]
#     while(y%2 == 0):
#         y = y//2
#     print('%d %d' %(y,l[i]//y))

# li = [11,22,33]
# new = list(map(lambda x:x+100 if(x<20) else x+200,li))
# print(li,new)


# class BinaryTreeNode():
#     def __init__(self,val,left = None,right = None):
#         self.val = val
#         self.left = left
#         self.right = right




# #-*- coding:utf-8 -*-
# class Solution:
#     def minNumberInRotateArray(self, rotateArray):
#         size = len(rotateArray)
#         if size==0:
#             return 0
#         if size==1:
#             return rotateArray[0]
#         if rotateArray[0]<rotateArray[1] and rotateArray[0]<rotateArray[-1]:
#             return rotateArray[0]
#         if rotateArray[-1]<rotateArray[0] and rotateArray[-1]<rotateArray[-2]:
#             return rotateArray[len(rotateArray)-1]
#         left,right = 0,size
#         while left<right:
#             pos = (left + right) // 2
#             if rotateArray[pos]<rotateArray[pos-1] and rotateArray[pos]<=rotateArray[pos+1]:
#                 return rotateArray[pos]
#             if rotateArray[pos]<=rotateArray[-1]:
#                 right = pos
#             else:
#                 left = pos+1
#
# li = list(range(100))
# li = li[34:]+li[:34]
# print(li)
# print(Solution().minNumberInRotateArray(li))


# # -*- coding:utf-8 -*-
# class Solution:
#     def reOrderArray(self, array):
#         if not array:
#             return
#         for i in range(len(array)//2):
#             for j in range(len(array)-i-1):
#                 if array[j]%2==0 and array[j+1]%2==1:
#                     array[j],array[j+1] = array[j+1],array[j]
#
#             print(array)
#
#
# Solution().reOrderArray(list(range(100)))
# # # -*- coding:utf-8 -*-
# # from collections import deque
# # class Solution:
# #     def reOrderArray(self, array):
# #         q = deque()
# #         for i in range(len(array)):
# #             if array[i]%2==0:
# #                 q.append(array[i])
# #             if array[len(array)-i-1]%2==1:
# #                 q.appendleft(array[len(array)-i-1])
# #         return q


# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:#输出链表的倒数第k个节点
#     def FindKthToTail(self, head, k):
#         p,count = head,0
#         while p:
#             p = p.next
#             count += 1
#         if k<count or count<=0:return
#         p = head
#         for i in range(count-k):
#             p = p.next
#         return p






# class Solution:              #反转链表
#     # 返回ListNode
#     def ReverseList(self, pHead):
#         if not pHead:return
#         if not pHead.next:
#             print pHead.val
#         left = pHead
#         p = left.next
#         while p:
#             right = p.next
#             p.next = left
#             left = p
#             p = right
#         pHead.next = None
#         pHead = left
#         return pHead




# # -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     # 返回ListNode
#     def ReverseList(self, pHead):
#         if not pHead or not pHead.next:
#             return pHead
#         last = None
#         while pHead:
#             temp = pHead.next
#             pHead.next = last
#             last = pHead
#             pHead = temp
#         return last
#
#
#
#
# pHead = ListNode(10)
# head1 = ListNode(20)
# pHead.next = head1
#
# Solution().ReverseList(pHead)


# -*- coding:utf-8 -*-
# class ListNode:    #合并两个有序链表
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     # 返回合并后列表
#     def Merge(self, pHead1, pHead2):
#         if not pHead1:return pHead2
#         if not pHead2:return pHead1
#         if pHead1.val<=pHead2.val:
#             cur = pHead1
#             head = cur
#             p1,p2 = pHead1.next,pHead2
#         else:
#             cur = pHead2
#             head = cur
#             p1,p2 = pHead1,pHead2.next
#
#         while p1 and p2:
#             if p1.val<=p2.val:
#                 cur.next = p1
#                 p1 = p1.next
#             else:
#                 cur.next = p2
#                 p2 = p2.next
#             cur = cur.next
#         if not p1:
#             cur.next = p2
#         else:
#             cur.next = p1
#         return head



# # -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# class Solution:
#     def HasSubtree(self, pRoot1, pRoot2):
#         if not pRoot2:return False
#         queue = [pRoot1]
#         while queue:
#             node = queue.pop(0)
#             if not node:continue
#             if node.val ==pRoot2.val:
#                 if self.Judge_two_BinaryTrees(node,pRoot2):
#                     return True
#             queue.append(node.left)
#             queue.append(node.right)
#
#         return False
#
#
#     def Judge_two_BinaryTrees(self,root1,root2):
#         if root1.val != root2.val or not root2:return False
#         queue1,queue2 = [root1],[root2]
#         while queue2:
#             node1,node2 = queue1.pop(0),queue2.pop(0)
#             if not node1:return False
#             if node1.val != node2.val:return False
#             if node2.left:
#                 queue2.append(node2.left)
#                 queue1.append(node1.left)
#             if node2.right:
#                 queue2.append(node2.right)
#                 queue1.append(node1.right)
#
#         if not queue2:
#             return True
#         else:
#             return False
#
#
# def list_create_BinaryTree(s,i):  #通过列表建立二叉树，如[8,8,7,9,2,'#','#','#','#',4,7]
#     if i<len(s):
#         if s[i] == '#':
#             return None
#         else:
#             root = TreeNode(s[i])
#             # print i
#             root.left = list_create_BinaryTree(s,2*i+1)
#             root.right = list_create_BinaryTree(s, 2*i + 2)
#             return  root
#
#     return None
#
# def hierar_traversal(root):
#     if not root:return
#     queue = [root]
#     while queue:
#         node = queue.pop(0)
#         print node.val
#         if node.left:
#             queue.append(node.left)
#         if node.right:
#             queue.append(node.right)
#
#
# s1 = [8,8,7,9,2,'#','#','#','#',4,7]
# s2 = [8,9,2]
# root1 = list_create_BinaryTree(s1,0)
# root2 = list_create_BinaryTree(s2,0)
#
# print Solution().Judge_two_BinaryTrees(root1,root2)
# hierar_traversal(root1)
# hierar_traversal(root2)





# # -*- coding:utf-8 -*-
# class Solution:                  #顺时针打印矩阵
#     # matrix类型为二维列表，需要返回列表
#     def printMatrix(self, matrix):
#         result = []
#         m,n = len(matrix),len(matrix[0])
#         book = [[False for i in range(n)] for j in range(m)]
#         i = 0
#         times = 0
#         x,y = 0,0
#         while True:
#             result.append(matrix[x][y])
#             book[x][y] = True
#             while times < 2:
#                 x1, y1 = self.get_next(x, y, i)
#                 if 0 <= x1 < m and 0 <= y1 < n and not book[x1][y1]:
#                     x, y = x1, y1
#                     times = 0
#                     break
#                 else:
#                     times += 1
#                     i += 1
#
#             if 0 <= x < m and 0 <= y < n and not book[x][y]:
#                 pass
#             else:
#                 break
#
#         return result
#
#
#
#     def get_next(self,x,y,i):   #i表示轮次除以4得出下一步方法
#         li = [(x,y+1),(x+1,y),(x,y-1),(x-1,y)]
#         return li[i%4][0],li[i%4][1]
#
#
#
# import numpy as np
# print Solution().printMatrix(np.array(list(range(1,17))).reshape(4,4).tolist())


# import numpy as np        #numpy数组  和 list之间的一些操作
# li = list(range(1,13))
# print li
# li2 = np.array(li)
# li.append(19)
# li2 = np.append(li2,19)
# print li,li2
# print type(li),type(li2)
# li2 = li2.reshape(3,4)
# li3 = li2.tolist()
# print li2,li3
# print type(li2),type(li3)


# # -*- coding:utf-8 -*-
# class Solution:                  #顺时针打印矩阵
#     # matrix类型为二维列表，需要返回列表
#     def printMatrix(self, matrix):
#         result = []
#         while matrix:
#             result.append(matrix.pop(0))
#             if not matrix:
#                 result = [j for i in result for j in i]
#                 return result
#             else:
#                 matrix = self.turn(matrix)
#
#     def turn(self,matrix):
#         B = []
#         m,n = len(matrix),len(matrix[0])
#         for j in range(n)[::-1]:
#             B.append([matrix[i][j]  for i in range(m)])
#
#         return B


# # -*- coding:utf-8 -*-
# class Solution:
#     def __init__(self):
#         self.stack = []
#     def push(self, node):
#         self.stack.append(node)
#
#     def pop(self):
#         return self.stack.pop()
#     def top(self):
#         return self.stack[-1]
#
#     def min(self):
#         min = self.stack[0]
#         for x in self.stack:
#             if min>x:
#                 min =x
#         return min


# # -*- coding:utf-8 -*-
# class Solution:
#     def IsPopOrder(self, pushV, popV):
#         if not pushV or len(pushV)!=len(popV):
#             return False
#         stack = []
#         for x in pushV:
#             stack.append(x)
#             while stack and stack[-1]==popV[0]:
#                 stack.pop()
#                 popV.pop(0)
#
#         if not stack:
#             return True
#         else:
#             return False

# print Solution().IsPopOrder([1,2,3,4,5],[4,5,3,2,1])


# # -*- coding:utf-8 -*-
# # class TreeNode:
# #     def __init__(self, x):
# #         self.val = x
# #         self.left = None
# #         self.right = None
# # class Solution:     #层次遍历二叉树hierarchical traversal
# #     # 返回从上到下每个节点值列表，例：[1,2,3]
# #     def PrintFromTopToBottom(self, root):
# #         if not root:return []
# #         queue,result = [root],[]
# #         while queue:
# #             node = queue.pop(0)
# #             result.append(node.val)
# #             if node.left:
# #                 queue.append(node.left)
# #             if node.right:
# #                 queue.append(node.right)
# #         return result
#
# # -*- coding:utf-8 -*-
# # class TreeNode:
# #     def __init__(self, x):
# #         self.val = x
# #         self.left = None
# #         self.right = None
# class Solution:
#     # 返回二维列表，内部每个列表表示找到的路径
#     def __init__(self):
#         self.stack = []
#         self.result = []
#
#     def FindPath(self, root, expectNumber):
#         if not root:return []
#         self.stack.append(root.val)
#         self.DFS(root,root.val,expectNumber)
#         return self.result
#
#     def DFS(self,cur,dis,expectNumber):
#         if not cur.left and not cur.right and dis==expectNumber:
#             self.result.append(list(self.stack))        #python必须用浅拷贝来添加列表，不然他们的对象引用是一样的(相当于起了别名)，之后改变其中之一，另一个也会改变
#             return
#         for x in [cur.left,cur.right]:
#             if x:
#                 self.stack.append(x.val)
#                 # print self.stack
#                 self.DFS(x,dis+x.val,expectNumber)
#                 self.stack.pop()
#
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# def list_create_BinaryTree(s,i):  #通过列表建立二叉树，如[8,8,7,9,2,'#','#','#','#',4,7]
#     if i<len(s):
#         if s[i] == '#':
#             return None
#         else:
#             root = TreeNode(s[i])
#             # print i
#             root.left = list_create_BinaryTree(s,2*i+1)
#             root.right = list_create_BinaryTree(s, 2*i + 2)
#             return  root
#
#     return None
#
#
# def print_BinaryTree(root):
#     if not root:return
#     queue = [root]
#     while queue:
#         node = queue.pop(0)
#         print node.val
#         if node.left:
#             queue.append(node.left)
#         if node.right:
#             queue.append(node.right)
#
# root = list_create_BinaryTree([10,5,12,4,7],0)
# # print_BinaryTree(root)
# print Solution().FindPath(root,22)









class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def list_create_BinaryTree(s,i):  #通过列表建立二叉树，如[8,8,7,9,2,'#','#','#','#',4,7]
    if i<len(s):
        if s[i] == '#':
            return None
        else:
            root = TreeNode(s[i])
            # print i
            root.left = list_create_BinaryTree(s,2*i+1)
            root.right = list_create_BinaryTree(s, 2*i + 2)
            return  root

    return None

def hierar_traversal(root):
    if not root:return
    queue = [root]
    while queue:
        node = queue.pop(0)
        print node.val
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)


class A(object):
    def __init__(self):
        pass
    def fun(self,expectNumber):
        self.expectNumber =expectNumber
        print self.expectNumber

# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        if not ss:return []
        ss = list(ss)
        length = len(ss)
        self.result = []
        self.get_permutation(ss,0,length)      #得到全排列 列表
        self.result = [''.join(x) for x in self.result]
        result = []
        for x in self.result:
            if x not in result:
                result.append(x)
        result = self.radix_sort(result,length)

        return result

    def get_permutation(self,li,m,n):    #m表示当前搜寻到的位置，n表示列表长度
        if m == n-1 :
            self.result.append(list(li))
        else:
            for i in range(m,n):
                li[i],li[m] = li[m],li[i]
                self.get_permutation(li,m+1,n)
                li[i],li[m] = li[m],li[i]

    def radix_sort(self,s,times):   #输入为一个列表，并且每个元素是一个整型列表
        for k in range(1,times+1):
            buckets = [[] for i in range(26)]
            for x in s:
                buckets[ord(x[-k])-97].append(x)
            s = [j for i in buckets for j in i]
        return s



# print Solution().radix_sort(["abc","bac","acb","cab","bca","cba"],3)



# #字符串转列表   列表转字符串
# print list('abcd')
# print ''.join(['a', 'b', 'c', 'd'])


# li = [['a','b','a','c','a'],['a'],['a']]
# li[0],li[1]=li[1],li[0]
# print li
from itertools import permutations
import random
it = permutations('abc')
# print map(''.join,it)               #通过itertools 生成全排列的迭代器对象，用map映射化字符串函数得到  字符串列表




# print sorted(['abc', 'acb', 'bac', 'bca', 'cab', 'cba'])




# # -*- coding:utf-8 -*-
# import itertools
# class Solution:
#     def Permutation(self, ss):
#         # write code here
#         if not ss:
#             return []
#         return sorted(list(set(map(''.join, itertools.permutations(ss)))))








# import itertools    #调用itertools极其简单方法
# #-*- coding:utf-8 -*-
# class Solution:
#     def Permutation(self, ss):
#         if not ss:return []
#         return sorted(list(set(map(''.join,itertools.permutations(ss)))))

# # -*- coding:utf-8 -*-
# class Solution:
#     def FindGreatestSumOfSubArray(self, array):
#         if not array:return
#         if len(array)==1:
#             return array[0]
#
#         left,right,Sum = 0,0,array[0]
#         for i in range(right+1,len(array)):
#             temp = left
#             for j in range(temp,i+1):
#                 if sum(array[j:i+1])>=Sum:
#                     left,right = j,i
#                     Sum = sum(array[j:right+1])
#
#         return Sum


# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        if not array:return
        dp = [array[0]]*len(array)
        for i in range(1,len(array)-1):
            dp[i]=max(dp[i-1],dp[i-1]+array[i],array[i])
        return dp[-1]


#


# print Solution().FindGreatestSumOfSubArray([1,3,-5,-3,5])
# print int('0xffffff',16)

#     # -*- coding:utf-8 -*-
# class Solution:
#     def FindGreatestSumOfSubArray(self, array):
#         if not array:return
#         if len(array)==1:
#             return array[0]
#
#         left,right,Sum = 0,0,array[0]
#         for i in range(right+1,len(array)):
#             temp = left
#             for j in range(temp,i+1):
#                 if sum(array[j:i+1])>=Sum:
#                     left,right = j,i
#                     Sum = sum(array[j:right+1])
#
#         return Sum



#sort,sorted 函数的一些用法
# L = [{1:5,3:4},{1:3,6:3},{1:1,2:4,5:6},{1:9}]
# def f2(a,b):
#     return a[1]-b[1],10
# def f1(a,b):
#     return a-b
# li = [12,4334,534,3,5]
# print sorted(li,cmp = f1)

# # -*- coding:utf-8 -*-
# class Solution:
#     def GetUglyNumber_Solution(self, index):
#         if index<=6:return index
#         result = [1]
#         u2,u3,u5 = [],[],[]
#         index_2,index_3,index_5 = 0,0,0
#         for i in range(2,index+1):
#             cur = min(2*result[index_2],3*result[index_3],5*result[index_5])
#             result.append(cur)
#             if cur%2 == 0:
#                 index_2 +=1
#                 u2.append(cur)
#             if cur%3 == 0:
#                 index_3 +=1
#                 u3.append(cur)
#             if cur%5 == 0:
#                 index_5 +=1
#                 u5.append(cur)
#
#         print u2
#         print u3
#         print u5
#
#         return result[-1]
#
# print Solution().GetUglyNumber_Solution(20)


# from collections import OrderedDict
# # -*- coding:utf-8 -*-
# class Solution:
#     def FirstNotRepeatingChar(self, s):
#         if not s:return -1
#         if len(s)==1:return s[0]
#         d = OrderedDict()
#         for i in range(len(s)):
#             if not d.get(s[i]):
#                 d[s[i]] = [1,i]
#             else:
#                 d[s[i]][0] +=1
#         print d
#         L = [{key:d[key]} for key in d.keys()]
#         print L
#         L.sort(key = lambda x:x.values()[0][0])
#         return L[0].values()[0][1]
#
#
#
# # print Solution().FirstNotRepeatingChar('google')

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:
#     def FindFirstCommonNode(self, pHead1, pHead2):
#         if not pHead1 or not pHead2: return
#         if id(pHead1) == id(pHead2):
#             return pHead1
#         elif not pHead1.next or not pHead2.next:
#             return
#         p2 = pHead2
#         while p2.next:
#             p2 = p2.next
#         p2.next = pHead2  # 将第二个链表首尾相连
#         # 找到相遇点，找不到说明两链表不想交
#         p, q = pHead1, pHead2
#         while p and q:
#             p = p.next
#             q = q.next.next
#             if id(p) == id(q): break
#
#         if id(p) != id(q): return
#         # 一个指针从pHead1出发，一个从相遇点出发,然后他们第一次相遇的地方就是交点
#         p1, p2 = pHead1, p
#         while p1 and p2:
#             if id(p1) == id(p2):
#                 return p1
#             p1, p2 = p1.next, p2.next
#
#
#
#
# # -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# class Solution(object):
#     def TreeDepth(self, pRoot):
#         if not pRoot:return 0
#         self.maxdepth = 1
#         self.DFS(pRoot,1)
#         return self.maxdepth
#
#     def DFS(self,cur,depth):
#         if not cur.left and not cur.right:
#             if self.maxdepth<depth:
#                 self.maxdepth = depth
#             return
#
#         for node in [cur.left,cur.right]:
#             if node:
#                 self.DFS(node,depth+1)
#
#
#
# def list_create_BinaryTree(s,i):
#     if i<len(s):
#         if s[i]=='#':
#             return None
#         else:
#             root = TreeNode(s[i])
#             root.left = list_create_BinaryTree(s,2*i+1)
#             root.right =list_create_BinaryTree(s,2*i+2)
#             return root
#
#     return
#
# def hierarchical_traversal(root):
#     if not root:return
#     queue = [root]
#     while queue:
#         node = queue.pop(0)
#         print node.val
#         if node.left:
#             queue.append(node.left)
#         if node.right:
#             queue.append(node.right)
#
#
# root = list_create_BinaryTree([8,9,2,3,'#',4],0)
# # hierarchical_traversal(root)
# print Solution().TreeDepth(root)





# from collections import Counter,OrderedDict
# class Solution:
#     # 返回[a,b] 其中ab是出现一次的两个数字
#     def FindNumsAppearOnce(self, array):
#         d = Counter(array)
#         for x in d:
#             print x,d[x]
#         for i in range(7):
#             print d[i]
#         print d.items()
#         print d.keys()
#         print d.values()
#         return d
# result = Solution().FindNumsAppearOnce([1,3,4,5,3,4,5,2,4])
# print result
# print dict(result)


# s1 = set([1,2,3,4])
# s2 = set([2,3,5,6])
# print s1|s2-(s1&s2)
# print s1.union(s2).difference(s1.intersection(s2))


# # -*- coding:utf-8 -*-
#输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
# class Solution:
#     def FindNumbersWithSum(self, array, tsum):
#         for x in array:
#             if x>tsum/2:
#                 break
#             if self.BinarySearch(array,tsum-x)!=None:
#                 return [x,tsum-x]
#         return []
#
#     def BinarySearch(self,array,a):
#         low,high = 0,len(array)
#         while low<high:
#             mid = (low+high)//2
#             if a==array[mid]:
#                 return mid
#             elif a<array[mid]:
#                 high = mid
#             else:
#                 low = mid+1
#         return None
# print Solution().FindNumbersWithSum([1,2,4,7,11,16],10)





# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        d = {}
        for x in duplication:
            if not d.get(x):
                d[x] =1
            else:
                duplication[0]=x
                return True
        return False



# #求排列数
# li = []
# stack = list(range(4))
# def get_permutation(m,n):
#     if m==n-1:
#         li.append(list(stack))
#     else:
#         for i in range(m,n):
#             stack[m],stack[i]= stack[i],stack[m]
#             get_permutation(m+1,n)
#             stack[m], stack[i] = stack[i], stack[m]
# get_permutation(0,4)
# print li

# W = [1,8,3,4,5,2]
# T = 10
# k = 0
# stack = []
# while stack or k<len(W):
#     while k<len(W):
#         if W[k]<=T:
#             stack.append(k)
#             T -= W[k]
#         k+=1
#     if T==0:
#         print stack
#     k = stack.pop()
#     T += W[k]
#     k += 1



##利用背包的思想输出组合数
# ss = []
# k,j = 0,0
# while ss or k<5:
#     while k<5:
#         ss.append(k)
#         k += 1
#     print ss
#     j +=1
#     k = ss.pop()
#     k +=1
# print j





# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        if not path or not matrix:return
        # 找到所有开始符合的点
        for i in range(rows):
            for j in range(cols):
                if matrix[i*cols+j]==path[0]:
                    if self.DFS(list(matrix),rows,cols,path[1:],i,j):
                        return True
        return False
    def DFS(self,matrix,rows,cols,path,i,j):
        if not path:
            return True
        matrix[i*cols+j] = '#'
        if i-1>=0 and matrix[(i-1)*cols+j]==path[0] and self.DFS(list(matrix),rows,cols,path[1:],i-1,j):
            return True
        elif i+1<rows and matrix[(i+1)*cols+j]==path[0] and self.DFS(matrix,rows,cols,path[1:],i+1,j):
            return True
        elif j+1<cols and matrix[i*cols+j+1]==path[0] and self.DFS(matrix,rows,cols,path[1:],i,j+1):
            return True
        elif j-1>=0 and matrix[i*cols+j-1]==path[0] and self.DFS(matrix,rows,cols,path[1:],i,j-1):
            return True
        else:
            return False
# # -*- coding:utf-8 -*-
# class Solution:
#     def hasPath(self, matrix, rows, cols, path):
#         if not path or not matrix:return
#         self.book = [[False for j in range(cols)] for i in range(rows)]
#         # 找到所有开始符合的点
#         for i in range(rows):
#             for j in range(cols):
#                 if matrix[i*cols+j]==path[0]:
#                     self.book[i][j] = True
#                     if self.DFS(0, (i,j), matrix,rows,cols, path):
#                         return True
#                     else:
#                         self.book[i][j] = False
#         return False
#
#     def DFS(self,i,cur,matrix,rows,cols,path):
#         if i==len(path)-1 and matrix[cur[0]*cols+cur[1]]==path[i]:
#             return True
#         if matrix[cur[0]*cols+cur[1]]==path[i] and i+1<len(path):
#             for x in [(cur[0]-1,cur[1]),(cur[0]+1,cur[1]),(cur[0],cur[1]-1),(cur[0],cur[1]+1)]:
#                 if 0<=x[0]<rows and 0<=x[1]<cols and not self.book[x[0]][x[1]]:
#                     self.book[x[0]][x[1]] = True
#                     if self.DFS(i+1,x,matrix,rows,cols,path):
#                         return True
#                     else:
#                         self.book[x[0]][x[1]] = False
#         return False


# matrix = ['a','b','c','e','s','f','c','s','a','d','e','e']
# path = ['bcced','abcb','bccee','abcceese']
# for p in path:
#     print Solution().hasPath(matrix,3,4,p)
# print Solution().hasPath("ABCEHJIGSFCSLOPQADEEMNOEADIDEJFMVCEIFGGS",5,8,"SLHECCEIDEJFGGFIE")
# # print Solution2().hasPath("ABCEHJIGSFCSLOPQADEEMNOEADIDEJFMVCEIFGGS",5,8,"SLHECCEIDEJFGGFIE")






# # -*- coding:utf-8 -*-
# class Solution:
#     def movingCount(self, threshold, rows, cols):
#         #可以直接广度优先加入所有符合的格子
#         count = 0
#         queue,book = [(0,0)],[[False for j in range(cols)] for i in range(rows)]
#         while queue:
#             node = queue.pop(0)
#             if not book[node[0]][node[1]]:
#                 count += 1
#                 book[node[0]][node[1]] = True
#             for nei in [(node[0],node[1]-1),(node[0],node[1]+1),(node[0]-1,node[1]),(node[0]+1,node[1])]:
#                 if 0<=nei[0]<rows and 0<=nei[1]<cols and not book[nei[0]][nei[1]]\
#                     and sum(map(int,str(nei[0])+str(nei[1])))<=threshold and nei not in queue:
#                     queue.append(nei)
#         return count
# print Solution().movingCount(15,20,20)





#实现分割长度为2n的数组，使两个子数组之和尽可能近
# import numpy as np
# import copy
# li = [11,8,9,2,15,7]
# dp = np.array([[False]*((sum(li)>>1)+1) for i in range(len(li)+1)])
# dp[0][0] = True              #dp[i][j]表示在前i个元素里取任意个元素能使其和为j
# dp2 = copy.deepcopy(dp)
# dp3 = copy.deepcopy(dp)
# print '----------------------------------'
# for k1 in range(1,len(li)+1):
#     for k2 in range(1,k1+1)[::-1]:
#         for s in range(1,(sum(li)>>1)+1):
#             if s>=li[k1-1] and dp[k2-1][s-li[k1-1]]:
#                 dp[k2][s] = True
#     print dp2
#
# for k in range(2,len(li)+1):
#     for s in range(1,(sum(li)>>1)+1):
#         if dp[k-1][s]:
#             dp[k][s] = True
# print 'last dp = \n{}'.format(dp)

# for k1 in range(1,len(li)+1):
#     for k2 in range(1,k1+1):    #此处应该用  range(1,k1+1)[::-1]  必须倒序，这样保证每一次更新都是用的上一次
#                                  # 在前k1个里面任取k2个元素之和的dp状态值，如果不倒序会使得dp结果可能会两次利用到同一个元素的和
#                                  #如li = [11,8,9,2,15,7]    会使dp[6][4]为真，因为两次用到了2
#         for s in range(1,(sum(li)>>1)+1):
#             if s>=li[k1-1] and dp2[k2-1][s-li[k1-1]]:
#                 dp2[k2][s] = True
#     print dp2
#
# for k in range(2,len(li)+1):
#     for s in range(1,(sum(li)>>1)+1):
#         if dp2[k-1][s]:
#             dp2[k][s] = True
# print 'last dp2 = \n{}'.format(dp2)



# for k in range(1,len(li)+1):
#     for s in range((sum(li)>>1)+1):
#         if s>=li[k-1]:
#             dp3[k][s] = dp3[k-1][s-li[k-1]] or dp3[k-1][s]
#         else:
#             dp3[k][s] = dp3[k-1][s]
# print dp3
#
# print '**************************************************'
# print dp[-1]
# print dp2[-1]
# print dp3[-1]
# print 'fdsfdsafasdff'
# print dp[-1]==dp3[-1]


G = {1:[2,3,4],
     2:[1,5],
     3:[1,5,6],
     4:[1,5],
     5:[2,3,4],
     6:[3]}

# G2 = dict((key,set(G[key])) for key in G)
# print G2
# def BFS(v):
#     visited = dict((key,False) for key in G)
#     queue = [v]
#     res = []
#     while queue:
#         v0 = queue.pop(0)
#         if not visited[v0]:
#             res.append(v0)
#             visited[v0] = True
#         for x in G[v0]:
#             if not visited[x]:
#                 queue.append(x)
#     return res

# def BFS_printbylayer(v,G):
#     visited = dict((key,False) for key in G)
#     current_nodes = [v]
#     visited[v] = True
#     while current_nodes:
#         print current_nodes
#         next_nodes = []
#         for node in current_nodes:
#             for nei in G[node]:
#                 if visited[nei]:
#                     continue
#                 visited[nei] = True
#                 next_nodes.append(nei)
#         current_nodes = list(next_nodes)
#
#
# BFS_printbylayer(1,G)












# coding: utf-8
'''

题意可以理解为：有没有可以通行的点，但牛牛到不了（输出-1）。如果没有这样的点，牛牛可能花费最多步数是多少。

思路：计算地图里，每个点的最短到达步数。找到到不了的点，或者步数最多的点。

1. 创建3个矩阵，size都是n*m的。分别是地图矩阵 mm、步数矩阵 sm、到达矩阵 am。详见代码里的注释。*也可以把3个矩阵放到一起。

2. 设置初始点为第一轮的探索点，更新 sm 里初始点的最短步数为0，更新 am 里初始点的到达状态为1。

3. 找到从探索点一步能到达的点，且这些点可以通行并没有到达过。更新这些点的 sm 和 am。并将这些点当作下一轮的探索点。

4. 循环第3步，直到没有新一轮的探索点了。

5. 从 sm 中可以得到正确答案。

'''

#此方法 很类似于层次打印二叉树，current_points表示当前层的节点，每一层的step一样
def main():
    line_1 = raw_input().split()

    n, m = int(line_1[0]), int(line_1[1])

    map_matrix = [raw_input() for i in range(n)]  # 地图矩阵，'.'表示可以通行，其他表示不可通行

    line_2 = raw_input().split()

    x0, y0 = int(line_2[0]), int(line_2[1])  # 开始点的坐标

    k = input()  # 共有k种行走方式

    alternative_steps = [[int(s) for s in raw_input().split()] for i in range(k)]  # 可以选择的行走方式

    step_matrix = [[-1] * m for i in range(n)]  # 步数矩阵，记录到达该点使用最短步数。初始是-1。

    arrived_matrix = [[0] * m for i in range(n)]  # 到达矩阵，记录是否已经到达过该点。初始是0，表示没有到达过，

    # 判断初始点是否是可达点

    if map_matrix[x0][y0] != '.':
        return -1

    # 初始点修改为已到达的点

    arrived_matrix[x0][y0] = 1

    # 初始点到达步数为0

    step_matrix[x0][y0] = 0

    current_points = [[x0, y0]]  # 第一轮所在的探索点（多个）

    while len(current_points) > 0:  # 如果当前探索点是0个，结束循环。

        next_points = []  # 下一轮的探索点（多个）

        for point in current_points:

            x, y = point[0], point[1]  # 一个探索点

            for step in alternative_steps:

                x_, y_ = x + step[0], y + step[1]  # 该探索点一步能到达的点

                if x_ < 0 or x_ >= n or y_ < 0 or y_ >= m:  # 检查是否越界

                    continue

                if map_matrix[x_][y_] != '.' or arrived_matrix[x_][y_] == 1:  # 检查该点是否可以通行，或者已经达到过

                    continue

                else:

                    step_matrix[x_][y_] = step_matrix[x][y] + 1  # 更新步数矩阵

                    arrived_matrix[x_][y_] = 1  # 更新到达矩阵

                    next_points.append([x_, y_])  # 该点添加到下一轮探索点里。

        current_points = next_points

    # 从步数矩阵中找到到不了的点，或者最大的步数。输出

    max_step = 0

    for i in range(n):

        for j in range(m):

            step = step_matrix[i][j]

            if step == -1 and map_matrix[i][j] == '.':
                return -1

            if step > max_step:
                max_step = step

    return max_step


# print main()



# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        if threshold<0:return 0
        #可以直接广度优先加入所有符合的格子
        book = [[False]*cols for i in range(rows)]
        current_points = [(0, 0)]
        book[0][0] = True
        count = 1
        while current_points:
            next_points = []
            for point in current_points:
                for t in [(1,0),(-1,0),(0,1),(0,-1)]:
                    next_x,next_y = point[0] + t[0],point[1] + t[1]
                    if next_x<0 or next_x>=rows or next_y<0 or next_y>=cols:
                        continue
                    if book[next_x][next_y] or sum(map(int,str(next_x)+str(next_y)))>threshold:
                        continue
                    book[next_x][next_y] = True
                    count += 1
                    next_points.append((next_x,next_y))
            current_points = list(next_points)
        return count

# # -*- coding:utf-8 -*-
# class Solution:
#     def movingCount(self, threshold, rows, cols):
#         if threshold<0:return 0
#         #可以直接广度优先加入所有符合的格子
#         count = 0
#         queue,book = [(0,0)],[[False for j in range(cols)] for i in range(rows)]
#         while queue:
#             node = queue.pop(0)
#             if not book[node[0]][node[1]]:
#                 count += 1
#                 book[node[0]][node[1]] = True
#             for nei in [(node[0],node[1]-1),(node[0],node[1]+1),(node[0]-1,node[1]),(node[0]+1,node[1])]:
#                 if 0<=nei[0]<rows and 0<=nei[1]<cols and not book[nei[0]][nei[1]]\
#                     and sum(map(int,str(nei[0])+str(nei[1])))<=threshold and nei not in queue:
#                     queue.append(nei)
#         return count

# print Solution().movingCount(5,10,10)



class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        if not pRoot:return []
        current_nodes,res = [pRoot],[]
        while current_nodes:
            res.append()
            next_nodes = []
            for node in current_nodes:
                if node.left:
                    next_nodes.append(node.left)
                if node.right:
                    next_nodes.append(node.right)
            current_nodes = list(next_nodes)
        return res

# class Solution:
#     # 返回二维列表[[1,2],[4,5]]
#     def Print(self, pRoot):
#         if not pRoot:return []
#         queue,tmp,res = [pRoot],[],[]
#         cur_last = pRoot
#         while queue:
#             node = queue.pop(0)
#             tmp.append(node.val)
#             if node.left:
#                 queue.append(node.left)
#             if node.right:
#                 queue.append(node.right)
#             if node==cur_last:
#                 res.append(tmp)
#                 cur_last = queue[-1] if queue else None
#                 tmp = []
#         return res





a ='fdsafsdafsadfsadfasdfds'
b = 'fasdfasdfdsfdsfad'
print(a)