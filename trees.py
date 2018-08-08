#决策树
#计算给定集的香农熵
from math import log

def calcShannonEnt(dataSset):
    numEntires=len(dataSset)
    labelCounts={}
    for featVec in dataSset:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntires
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
