import FuzzyClustering
import numpy as np
import matplotlib.pyplot as plt

'''数据及参数设置'''
data = np.arange(0,600).reshape(150,4)
cluster_n = 3

'''模糊聚类算法'''
U,center,fcm_obj_fcn = FuzzyClustering.fcm(data,cluster_n)

'''画图'''
plt.plot(fcm_obj_fcn)
plt.show()