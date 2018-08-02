import numpy as np
import math

# print(np.eye(4,3))
# x = np.zeros((4,3))
# print(x)

# for i in range(0, 10):
#     print(i)
# print(np.eye(2) * 0.01)
# F = np.matrix('1 0 0.1 0; 0 1 0 0.1; 0 0 1 0; 0 0 0 1')
# print(F)

# print(np.eye(2))

# 2x2
a = np.matrix('1 2; 2 0');
# 2x1
b = np.matrix('2;3');
# 1x2
c = np.matrix('5 9');

print(a * b * c)
print((a * b) * c)
print(a * (b * c))
