# # import string
# def dis(s,t):
#     sum = 0
#     for i in range(len(s)):
#         if(s[i]!=t[i]):
#             sum += 1
#     return sum
#
# s= str(raw_input())
# s = s.strip()
# t= str(raw_input())
# t = t.strip()
# last_sum = 0
# for i in range(len(t)):
#     for j in range(i,i+len(s)-len(t)+1):
#         if t[i]!=s[j]:
#             last_sum +=1
# print(last_sum)

s = str(raw_input())
sum =[0,0,0,0,0,0,0,0,0,0]
for i in range(len(s)):
    sum[int(s[i])] +=1

min_num = min(sum)
min_index =sum.index(min(sum))
if min_num == 0 and min_index!= 0:
    result =min_index



print result