import numpy as np

# 打开训练数据集
def file(filename):
    dataMat= []; labelMat = []
    fr = open(filename)
    numFeat = len(fr.readline().split('\t')) - 1
    for line in fr.readlines():
        lineArr = []
        curArr = line.strip().split('\t')
        for i in range(numFeat) :
            lineArr.append(float(curArr[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curArr[-1]))
    dataArr = np.mat(dataMat)
    labelArr = np.mat(labelMat)
    return dataArr, labelArr

#定义sigmoid函数
def sigmoid(z):
    return 1/(1+ np.exp(-z))

#梯度下降求解
def gradAscent(dataArr, labelArr,alpha = 0.02):
    m,n = np.shape(dataArr)
    w = np.ones((n,1))
    maxcycle = 500
    for i in range(maxcycle):
        z = dataArr * w
        h = sigmoid(z)
        error = labelArr.T - h
        w =  w + alpha *  dataArr.T * error
    return w

dataArr, labelArr = file('horseColicTest.txt')
w = gradAscent(dataArr,labelArr)

#打开测试数据：
def fileTest(filename):
    fr = open(filename)
    numFeat = len(fr.readline().split('\t')) - 1
    testMat =[]
    for line in fr.readlines():
        lineArr = []
        curArr = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curArr[i]))
            testMat.append(lineArr)
        testArr = np.mat(testMat)
    return testArr

#计算label:
testArr = fileTest('horseColicTest.txt')
Y = sigmoid(testArr * w)
print(Y)