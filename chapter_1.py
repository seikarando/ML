import numpy as np

a = np.random.rand(4, 4)  # 随机数组
randMat = np.mat(a)  # 随机数组产生矩阵
invRandMat = randMat.I  # 求逆
a = randMat * invRandMat  # 单位矩阵
np.eye(4)  # 单位矩阵
# randMat=np.mat(np.random.rand(4,4))
print(a)
print(randMat)

