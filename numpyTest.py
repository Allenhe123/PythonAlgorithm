import numpy as np
import math

'''
Python中提供了list容器，可以当作数组使用。但列表中的元素可以是任何对象，因此列表中保存的是对象的指针，这样一来，为了保存一个简单的列表[1,2,3]。就需要三个指针和三个整数对象。
对于数值运算来说，这种结构显然不够高效。
Python虽然也提供了array模块，但其只支持一维数组，不支持多维数组，也没有各种运算函数。因而不适合数值运算。
NumPy的出现弥补了这些不足。
'''

#数组创建
a = np.array([2, 3, 4])
print(a, a.dtype)

b = np.array([[1.0, 2.0], [3.0, 4.0]])
print(b, b.dtype)

c = np.array([2, 3, 4], dtype=float)
print(c, c.dtype)

d = np.array([[1, 2], [3, 4]], dtype=complex)
print(d, d.dtype)

print( np.arange(0, 7, 1, dtype=np.int16) )   # 0为起点，间隔为1时可缺省(引起歧义下不可缺省)

print( np.eye(3,4, dtype = np.int16) )        # 3行，4列，对角线全1，指定数据类型

print( np.ones((3,4), dtype = np.int16) )   # 3行，4列，全1，指定数据类型

print( np.ones((2,3,4), dtype = np.int16) )   # 2页，3行，4列，全1，指定数据类型

print( np.zeros((2,3,4)) )

print( np.linspace(-1,2,5) ) # 起点为-1，终点为2，取5个点

print( np.random.randint(0, 3, (2, 3)) ) # 大于等于0，小于3，2行3列的随机整数

## 类型转换
print( float(1) )
print( int(1.0) )
print( bool(2) )
print( float(True) )

'''
数组输出:
从左到右，从上向下
一维数组打印成行，二维数组打印成矩阵，三维数组打印成矩阵列表
'''
print( np.arange(1,6,2) )
print( np.arange(12).reshape(3,4) ) # 可以改变输出形状
print( np.arange(24).reshape(2,3,4) )# 2页，3行，4页

# 基本运算
## 元素级运算
a = np.array([1,2,3,4])
b = np.arange(4)
print (a, b)
print (a - b)
print (a * b)
print (a ** 2)
print (2 * np.sin(a))
print (a > 2)
print (np.exp(a)) # 指数


## 矩阵运算（二维数组）
a = np.array([[1,2],[3,4]])      # 2行2列
b = np.arange(6).reshape((2,-1)) # 2行3列
print( a, b )
print( a.dot(b) ) # 2行3列

## 非数组运算，调用方法
a = np.random.randint(0, 5, (2, 3))
print( a )
print( a.sum(), a.sum(axis=1), a.sum(0) )       # axis用于指定运算轴（默认全部，可指定0或1）
print( a.min(), a.max(axis=1), a.mean(axis=1) ) # axis = 0: 按列计算，axis = 1: 按行计算
print( a.cumsum(1) )                          # 按行计算累积和


## 一维数组
print('###############')
a = np.arange(0, 10, 1) ** 2
print (a)
print (a[0], a[2], a[-1], a[-2])   # 索引从0开始，-1表示最后一个索引
print (a[2:5], a[-5:-1])           # 包括起点，不包括终点
a[-1] = 100;                       # 赋值
print(a)
a[1:4] = 100;                      # 批量赋值
print (a)
a[:6:2] = -100;                    # 从开始到第6个索引，每隔一个元素（步长=2）赋值
print (a)
print (a[: :-1])                   # 将a逆序输出，a本身未发生改变
print (a)
b = [np.sqrt(np.abs(i)) for i in a]; print (b)    # 通过遍历赋值


## 多维数组
print('----------------------------------------')
a = np.arange(0,20).reshape((4,5))
print (a, a[2,3], a[:,1], a[1:4,2], a[1:3,:])
print('%%%%')
print (a[-1]) # 相当于a[-1,:],即索引少于轴数时，确实的索引默认为整个切片

print('------------------------------')
b = np.arange(0,24).reshape((2,3,4))
print (b, b[1]) # 相当于b[1,:,:] 和b[1,...]
print ('-------------------')
for row in a:
    print (row) # 遍历以第一个轴为基础

print('--------------------------------------')
a = np.floor( 10 * np.random.random( (3, 4)) )
print (a, a.shape)                                # 输出a的形状
print (a.ravel())                                 # 输出平坦化后的a（a本身不改变）
a.shape = (6,2); print (a)                        # 改变a的形状
print (a.transpose()) # 输出a的转置

print('-----------------------------------------')
## 补充：reshape和resize
a = np.array([[1, 2, 3], [4, 5, 6]])
b = a
a.reshape((3,2))        # 不改变数组本身的形状
print (a)
b.resize((3,2))         # 改变数组本身形状
print (b)
