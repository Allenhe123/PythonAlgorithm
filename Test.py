import numpy as np
import math

from functools import reduce

# print(np.eye(4,3))
# x = np.zeros((4,3))
# print(x)

# for i in range(0, 10):
#     print(i)
# print(np.eye(2) * 0.01)
# F = np.matrix('1 0 0.1 0; 0 1 0 0.1; 0 0 1 0; 0 0 0 1')
# print(F)

# print(np.eye(6))

# 2x2
# a = np.matrix('1 2; 2 0');
# # 2x1
# b = np.matrix('2;3');
# # 1x2
# c = np.matrix('5 9');
#
# print(a * b * c)
# print((a * b) * c)
# print(a * (b * c))

# a = 0.999
# b = 0.5
# print(math.pow(a, 20))
# print(math.pow(b, 20))

'''
在 Python3 中，reduce() 函数已经被从全局名字空间里移除了，它现在被放置在 fucntools 模块里，
如果想要使用它，则需要通过引入 functools 模块来调用 reduce() 函数：
'''
# reducevalue1 = reduce(lambda x,y : x + y, [1,2,3,4,5], 1)
# print (reducevalue1)
#
# reducevalue2 = reduce(lambda x,y : x + y, [1,2,3,4,5])
# print (reducevalue2)


a = [8, 1, 2, 10, 9, 6, 7, 8, 0, 4, 19, 22, 3]
maxv = 0
sndMaxv = 0
maxIdx = 0
sndMaxIdx = 0
for i in range(len(a)):
    if a[i] > maxv:
        sndMaxv = maxv
        sndMaxIdx = maxIdx
        maxv = a[i]
        maxIdx = i

print(maxv, maxIdx, sndMaxv, sndMaxIdx)