import numpy as np
import random
import math
import ClusteringIteration
import ClusterAidedComputing

def fcm(data,cluster_n,m = 2,max_iter = 1000,e = 0.00001 ,printOn = 1):
    data = np.array(data)
    obj_fcn = np.zeros(max_iter)

    # 随机初始化聚类中心
    center = np.random.rand(cluster_n,data.shape[1])
    
    # 初始化隶属度矩阵
    dist = ClusterAidedComputing.distfcm(data,center)
    U = np.zeros((center.shape[0],data.shape[0]))
    U = ClusterAidedComputing.tmp(dist)
    
    # 主循环
    for i in range(max_iter):
        U,center,obj_fcn[i] = ClusteringIteration.stepfcm(data,U,cluster_n,m)
        if printOn == 1:
            print("FCM第",i,"次迭代的目标函数值为:",obj_fcn[i])
        if i > 0:
            if abs(obj_fcn[i] - obj_fcn[i-1]) < e :
                break
            else:
                continue
        else:
            continue
    return U,center,obj_fcn
