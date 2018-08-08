# kNN k-近邻算法
import numpy as np
import operator  # 运算符模块
from imp import reload  # 没法正常下载


# 2.1 k-临近算法
# 2.2 示例：使用k-近邻算法改进约会网站的配对效果
# 2.3 手写识别系统
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 记住外面还有个[]
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()


# print(group)
# print(labels)

# tile测试 tile(矩阵，(纵，横））
# a = np.tile([1, 3], (4, 1))
# print(a)
# sqa = a ** 2
# print(sqa)

# range用法
# range(start, stop[, step])
# start: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）;
# stop: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
# step：步长，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)

# k-近邻算法
def classify0(inX, dataSet, labels, k):  # inX用于分类的输入向量
    dataSetSize = dataSet.shape[0]  # shape[0]=第一维度长度
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1：列  axis=0：行
    distances = sqDistances ** 0.5  # 欧氏距离公式sqrt(a^2+b^2)
    sortedDistIndicies = distances.argsort()  # 将distances中的元素从小到大排列，提取其对应的index，
    # 然后输出到sortedDistIndicies
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # key：关键词
    # 排序的参照，True：降序，False：升序
    return sortedClassCount[0][0]


# get()函数作用
# 以classCount.get(voteIlabel,0)为例：
# classCount.get(voteIlabel,0)返回字典classCount中voteIlabel元素对应的值,若无，则进行初始化
# 若不存在voteIlabel，则字典classCount中生成voteIlabel元素，并使其对应的数字为0，即
# classCount = {voteIlabel：0}
# 此时classCount.get(voteIlabel,0)作用是检测并生成新元素，***括号中的0只用作初始化，之后再无作用***
# 当字典中有voteIlabel元素时，classCount.get(voteIlabel,0)作用是返回该元素对应的值，即0
# 以书中代码为例：
# classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1；
# 初始化classCount = {}时，此时输入classCount，输出为：
# classCount = {}
# 当第一次遇到新的label时，将新的label添加到字典classCount，并初始化其对应数值为0
# 然后+1，即该label已经出现过一次，此时输入classCount，输出为：
# classCount = {voteIlabel：1}
# 当第二次遇到同一个label时，classCount.get(voteIlabel,0)返回对应的数值（此时括号内的0不起作用，
# 因为已经初始化过了），然后+1，此时输入classCount，输出为：
# classCount = {voteIlabel：2}
# 可以看出，+1是每一次都会起作用的,因为不管遇到字典内已经存在的或者不存在的，都需要把这个元素记录下来

# iteritems()函数作用：
# 以书中classCount.iteritems()为例，作用是将字典classCount分解为元组列表
# 若classCount = {‘A’：1，‘B’：2，‘C’：3}
# 则分为
# [‘A’,’B’,’C’] 与 [1, 2, 3]两组
# itemgetter()函数作用：
# 结合4，以书中operator.itemgetter(1)为例，作用是读取元组iteritems内的第2列，即字典classCount = {‘A’：1，‘B’：2，‘C’：3}中的[1, 2, 3]
print(classify0([0, 0], group, labels, 3))


# 2.2 示例：使用k-近邻算法改进约会网站的配对效果
# 2.2.1 准备数据：从文本文件中解析数据
# 将文本记录转换为numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 列表长度
    returnMat = np.zeros((numberOfLines, 3))  # 创建返回的矩阵
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():  # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
        line = line.strip()  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        listFromLine = line.split('\t')  # split() 通过指定分隔符对字符串进行切片
        returnMat[index, :] = listFromLine[0:3]  # '0:3':0,1,2
        classLabelVector.append(int(listFromLine[-1]))  # 在列表末尾添加新的对象。
        index += 1
    return returnMat, classLabelVector  # 这个缩进我艹了他了


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels)

# 2.2.2 分析数据：使用matplotlib创建散点图

import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 10))  # 建立画板

fig.add_subplot(234)  # 添加一个子图 将画布分割成2行2列，图像画在从左到右从上到下的第4块
idx1 = np.array(datingLabels) == 1
idx2 = np.array(datingLabels) == 2
idx3 = np.array(datingLabels) == 3
plt.scatter(datingDataMat[idx1, 0], datingDataMat[idx1, 1], marker='+', s=20, c='b', label='Don’t Like')
plt.scatter(datingDataMat[idx2, 0], datingDataMat[idx2, 1], marker='o', s=20, c='m', label='Small Doses')
plt.scatter(datingDataMat[idx3, 0], datingDataMat[idx3, 1], marker='x', s=20, c='c', label='Large Doses')
# marker:形状 label：标签 c：color s：size
plt.legend(loc='upper left')

fig.add_subplot(235)
plt.scatter(datingDataMat[idx1, 1], datingDataMat[idx1, 2], marker='+', s=20, c='b', label='Don’t Like')
plt.scatter(datingDataMat[idx2, 1], datingDataMat[idx2, 2], marker='o', s=20, c='m', label='Small Doses')
plt.scatter(datingDataMat[idx3, 1], datingDataMat[idx3, 2], marker='x', s=20, c='c', label='Large Doses')
plt.legend(loc='upper left')

fig.add_subplot(236)
plt.scatter(datingDataMat[idx1, 0], datingDataMat[idx1, 2], marker='+', s=20, c='b', label='Don’t Like')
plt.scatter(datingDataMat[idx2, 0], datingDataMat[idx2, 2], marker='o', s=20, c='m', label='Small Doses')
plt.scatter(datingDataMat[idx3, 0], datingDataMat[idx3, 2], marker='x', s=20, c='c', label='Large Doses')
plt.legend(loc='upper left')

fig.add_subplot(231)
plt.scatter(datingDataMat[:, 1], datingDataMat
[:, 2])
plt.ylabel('Kilogram of ice cream per week')
plt.xlabel('Percentage of time spent playing games')

fig.add_subplot(232)
plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 20 * np.array(datingLabels),
            20 * np.array(datingLabels))  # 第23行 #TODO: 不懂

fig.add_subplot(233)
plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 20 * np.array(datingLabels),
            20 * np.array(datingLabels))  # 第12行

plt.show()


# 2.2.3归一化特征值

def autoNorm(dataSet):
    # dataSet=dataSet.astype('float64')
    minVals = dataSet.min(0)  # (0)从列选取最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))  # shape:转化为(行，列)
    m = dataSet.shape[0]  # 0:行 1：列
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # print(dataSet.dtype)
    return normDataSet, ranges, minVals


normMat, ranges, minVals = autoNorm(datingDataMat)
print('归一化后特征值：')
print(normMat)
print('范围：')
print(ranges)
print('最小值：')
print(minVals)


# 2.2.4测试算法：作为完整程序验证分类器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # m=1000
    numTestVecs = int(m * hoRatio)  # numTestVecs=100
    errorCount = 0.0
    for i in range(numTestVecs):  # 900个做测试 100个做检测
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:%d, the real answer is %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is:%f" % (errorCount / float(numTestVecs)))


datingClassTest()


#  2.2.5 使用算法：构建完整可用系统

# def classifyPerson():
#     resultList = ['not at all', 'in small dose', 'in large doses']
#     ffMiles = float(input("frequent flier miles earned per year?"))
#     percentRate = float(input("frequent of time spent playing video games?"))
#     iceCream = float(input("liters of ice cream consumed per year?"))
#     datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
#     normMat, ranges, minVals = autoNorm(datingDataMat)
#     inArr = np.array([ffMiles, percentRate, iceCream])
#     classfierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
#     print("you probably like this person:", resultList[classfierResult - 1])


# classifyPerson()

# 2.3 手写识别系统

# 将图像转化为向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))  # 两个括号
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 2.3.2 手写数字识别系统的测试代码
from os import listdir


def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')#返回指定的文件夹包含的文件或文件夹的名字的列表。
    # 这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 选取第0个分片 0_0.txt -> 0_0&txt
        classNumStr = int(fileStr.split('_')[0])  # 数字的名字
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUndertest = img2vector('trainingDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUndertest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real number is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is %f" % (errorCount / float(mTest)))
handWritingClassTest()

