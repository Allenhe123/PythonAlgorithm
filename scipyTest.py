import numpy as np
import math
import scipy

'''
based on the effective data structures of numpy, scipy provides the senior algorithm applications.
数值分析算法库： 矩阵运算，线性代数，最优化方法，聚类，空间运算，FFT

cluster - 层次聚类(cluster.hierarchy)   矢量量化/K均值（cluster.vq）
constants - 物理和数学常量   转换方法
fftpack - 离散傅里叶变换
integrate - 积分例程
interpolate - 插值（线性，三次方。。等等）
io - 数据输入输出
linalg - 采用优化BLAS和APACK库的线性代数函数
maxentropy - 最大熵模型函数
ndimage - n维图像工具包
odr - 正交距离回归
optimize - 最优化（寻找极小值和方程的根）
signal - 信号处理
sparse - 稀疏矩阵
spatial - 空间数据结构和算法
special - 特殊数学函数如贝塞尔函数，雅可比函数
stats - 统计学工具包
'''

print(scipy.dot is np.dot)