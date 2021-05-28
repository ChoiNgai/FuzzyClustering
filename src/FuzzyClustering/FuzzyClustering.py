import numpy as np
import random
import math
from FuzzyClustering import ClusteringIteration
from FuzzyClustering import ClusterAidedComputing


'''模糊C均值聚类算法（FCM）'''
def fcm(data,cluster_n,m = 2,max_iter = 1000,e = 0.00001,printOn = 1):
    data = np.array(data)
    obj_fcn = np.zeros(max_iter)

    # 随机初始化聚类中心(并根据初始聚类中心生成初始隶属度矩阵)
    U,center = ClusterAidedComputing.initcenter(data,cluster_n)
    
    # 主循环
    for i in range(max_iter):
        U,center,obj_fcn[i] = ClusteringIteration.stepfcm(data,U,cluster_n,m)
        if printOn == 1:
            print("FCM第",i,"次迭代的目标函数值为:",obj_fcn[i])
        if i > 0:
            if abs(obj_fcn[i] - obj_fcn[i-1]) < e :
                break
    return U,center,obj_fcn


'''极大熵模糊聚类算法（MEC）'''
def mec(data,cluster_n,gamma=0.01,max_iter = 1000,e = 0.00001,printOn = 1):
    data = np.array(data)
    obj_fcn = np.zeros(max_iter)

    # 随机初始化聚类中心(并根据初始聚类中心生成初始隶属度矩阵)
    U,center = ClusterAidedComputing.initcenter(data,cluster_n)
    
    # 主循环
    for i in range(max_iter):
        U,center,obj_fcn[i] = ClusteringIteration.stepmec(data,U,cluster_n,gamma)
        if printOn == 1:
            print("MEC第",i,"次迭代的目标函数值为:",obj_fcn[i])
        if i > 0:
            if abs(obj_fcn[i] - obj_fcn[i-1]) < e :
                break
    return U,center,obj_fcn


'''基于高斯核的模糊C均值聚类算法（KFCM）'''
def kfcm(data,cluster_n,sigma=2,m=2,lamda=0.1,max_iter = 1000,e = 0.00001,printOn = 1):
    data = np.array(data)
    obj_fcn = np.zeros(max_iter)

    # 随机初始化聚类中心(并根据初始聚类中心生成初始隶属度矩阵)
    U,center = ClusterAidedComputing.initcenter(data,cluster_n)
    
    # 主循环
    for i in range(max_iter):
        U,center,obj_fcn[i] = ClusteringIteration.stepkfcm(data,U,cluster_n,sigma,m)
        if printOn == 1:
            print("KFCM第",i,"次迭代的目标函数值为:",obj_fcn[i])
        if i > 0:
            if abs(obj_fcn[i] - obj_fcn[i-1]) < e :
                break
    return U,center,obj_fcn


'''半监督模糊C均值聚类算法（SFCM）'''
def sfcm(data,cluster_n,label,m = 2,max_iter = 1000,e = 0.00001,alpha=5,printOn = 1):
    data = np.array(data)
    obj_fcn = np.zeros(max_iter)

    # 随机初始化聚类中心(并根据初始聚类中心生成初始隶属度矩阵)
    U,center = ClusterAidedComputing.initcenter(data,cluster_n)
    
    # 根据类别标签信息生产先验隶属度矩阵
    F = ClusterAidedComputing.PriorMembership(label,U)

    # 主循环
    for i in range(max_iter):
        U,center,obj_fcn[i] = ClusteringIteration.stepsfcm(data,U,cluster_n,m,F,alpha)
        if printOn == 1:
            print("SFCM第",i,"次迭代的目标函数值为:",obj_fcn[i])
        if i > 0:
            if abs(obj_fcn[i] - obj_fcn[i-1]) < e :
                break
    return U,center,obj_fcn

'''基于信息熵的半监督模糊C均值聚类算法（eSFCM）'''
def esfcm(data,cluster_n,label,max_iter = 1000,e = 0.00001,lamda=1,printOn = 1):
    data = np.array(data)
    obj_fcn = np.zeros(max_iter)

    # 随机初始化聚类中心(并根据初始聚类中心生成初始隶属度矩阵)
    U,center = ClusterAidedComputing.initcenter(data,cluster_n)
    
    # 根据类别标签信息生产先验隶属度矩阵
    F = ClusterAidedComputing.PriorMembership(label,U)

    # 主循环
    for i in range(max_iter):
        U,center,obj_fcn[i] = ClusteringIteration.stepesfcm(data,U,cluster_n,F,lamda)
        if printOn == 1:
            print("eSFCM第",i,"次迭代的目标函数值为:",obj_fcn[i])
        if i > 0:
            if abs(obj_fcn[i] - obj_fcn[i-1]) < e :
                break
    return U,center,obj_fcn


'''基于度量学习与信息熵的半监督模糊C均值聚类算法（SMUC）'''
def smuc(data,cluster_n,label,max_iter = 1000,e = 0.00001,lamda=1,printOn = 1):
    data = np.array(data)
    obj_fcn = np.zeros(max_iter)

    # 随机初始化聚类中心(并根据初始聚类中心生成初始隶属度矩阵)
    U,center = ClusterAidedComputing.initcenter(data,cluster_n)
    
    # 根据类别标签信息生产先验隶属度矩阵
    F = ClusterAidedComputing.PriorMembership(label,U)

    # 主循环
    for i in range(max_iter):
        U,center,obj_fcn[i] = ClusteringIteration.stepsmuc(data,U,cluster_n,F,lamda)
        if printOn == 1:
            print("eSFCM第",i,"次迭代的目标函数值为:",obj_fcn[i])
        if i > 0:
            if abs(obj_fcn[i] - obj_fcn[i-1]) < e :
                break
    return U,center,obj_fcn
