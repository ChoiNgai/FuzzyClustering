import numpy as np

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
