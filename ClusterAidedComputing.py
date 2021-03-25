'''聚类中的一些计算常用函数'''
import numpy as np
import math
import random

'''随机初始化聚类中心和隶属度矩阵'''
def initcenter(data,cluster_n):
    sample_index = random.sample(range(0,data.shape[0]),cluster_n) 
    center = data[sample_index]
    # (根据初始聚类中心)初始化隶属度矩阵
    dist = distfcm(data,center)**2
    U = np.zeros((center.shape[0],data.shape[0]))
    U = tmp(dist)   #统一根据FCM的迭代式初始化聚类中心
    return U,center

'''欧式距离函数'''
    # 输入：
    # center ——聚类中心
    # data  ——样本
    # 输出：
    # dist ——距离矩阵
def distfcm(data,center):
    dist = np.zeros( (center.shape[0],data.shape[0]))
    for i in range(center.shape[0]):
        for j in range(data.shape[0]):
            dist[i][j] = np.sum( abs( data[j] - center[i] ) )
    return dist
    
'''矩阵除以每一列之和（类似softmax函数）'''    
def tmp(x):
    # 计算每行的最大值
    new_x = np.zeros((x.shape[0],x.shape[1]))
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            new_x[i][j] = x[i][j] / np.sum(x[:,j])
    return new_x

'''计算聚类中心'''
def centercompute(data,U):
    cluster_n = U.shape[0]
    mf = np.zeros((1,cluster_n))
    for i in range(0,cluster_n):
        mf[0][i] = np.sum(U[i])
    mf = np.matmul( np.ones((data.shape[1],1)),mf)
    mf = mf.T 
    center = np.matmul(U,data) / mf
    return center

'''根据类别标签生产先验隶属度矩阵'''
def PriorMembership(label,U):
    F = np.zeros(U.shape)
    for i in range(len(label)):
        F[int(label[i])-1][i] = 1
    return F

'''矩阵逐元素指数次方 exp(dist./gamma)  '''
def MatrixElementPower(dist,gamma):
    dist_exp = np.array(dist)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist_exp[i][j] = math.exp(-dist[i][j]**2/gamma)
    return dist_exp

'''矩阵逐元素log'''
def MatrixElementLog(U):
    U_log = np.array(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if U[i][j] != 0 :
                U_log[i][j] = math.log(U[i][j])
            else:
                U_log[i][j] = U[i][j]
    return U_log