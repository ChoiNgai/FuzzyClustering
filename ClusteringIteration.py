'''聚类中心矩阵和隶属度矩阵的迭代式'''
import numpy as np
import ClusterAidedComputing

'''FCM 迭代式'''
def stepfcm(data,U,cluster_n,m):
    mf = U**m
    center = ClusterAidedComputing.centercompute(data,mf)
    dist  = ClusterAidedComputing.distfcm(data,center)
    U = ClusterAidedComputing.tmp( dist**(-2/(m-1)) )
    obj_fcn = np.sum( (dist**2)*U )
    return U,center,obj_fcn

'''MEC 迭代式'''
def stepmec(data,U,cluster_n,gamma):
    center = ClusterAidedComputing.centercompute(data,U)
    dist  = ClusterAidedComputing.distfcm(data,center)
    dist_exp = ClusterAidedComputing.MatrixElementPower(dist,gamma)   #矩阵逐个元素平方取负数再除以gamma
    U = ClusterAidedComputing.tmp( dist_exp )
    obj_fcn = np.sum( (dist**2)*U ) + gamma * np.sum( U * ClusterAidedComputing.MatrixElementLog(U)  )
    return U,center,obj_fcn

'''SFCM 迭代式'''
def stepsfcm(data,U,cluster_n,m,F,alpha):
    mf = U**m
    center = ClusterAidedComputing.centercompute(data,mf)
    dist  = ClusterAidedComputing.distfcm(data,center)
    U_fcm = ClusterAidedComputing.tmp( dist**(-2/(m-1)) )
    U_2 = (alpha/(alpha+1)) * U_fcm * np.matmul( np.ones((cluster_n,1)) ,np.sum(F,0).reshape(1,data.shape[0])) 
    U =  U_fcm + (alpha/(1+alpha)) * F - U_2 
    obj_fcn = np.sum( (dist**2)*U )
    return U,center,obj_fcn

'''eSFCM 迭代式'''
def stepesfcm(data,U,cluster_n,F,lamda):
    center = ClusterAidedComputing.centercompute(data,U)
    dist  = ClusterAidedComputing.distfcm(data,center)
    dist_exp = ClusterAidedComputing.MatrixElementPower(dist,1/lamda)
    U = ClusterAidedComputing.tmp( dist_exp )
    U = F + U*(1-F)
    obj_fcn = np.sum( (dist**2)*U )
    return U,center,obj_fcn