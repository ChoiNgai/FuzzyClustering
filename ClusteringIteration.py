import numpy as np
import ClusterAidedComputing

def stepfcm(data,U,cluster_n,m):
    mf = U**m
    center = ClusterAidedComputing.centercompute(data,mf)
    dist  = ClusterAidedComputing.distfcm(data,center)
    obj_fcn = np.sum( (dist**2)*mf )
    U = ClusterAidedComputing.tmp( dist**(-2/(m-1)) )
    return U,center,obj_fcn