import FuzzyClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


'''数据及参数设置'''
data_route = r'dataset\iris.csv'
label_route = r'dataset\irislabel.csv'

with open(data_route,encoding = 'utf-8') as f:
    data = np.loadtxt(f,delimiter = ",")
with open(label_route,encoding = 'utf-8') as f:
    label = np.loadtxt(f,delimiter = ",")

cluster_n = int(np.max(label))
data = ( data - np.min(data,axis=0)) / (np.max(data,axis=0) - np.min(data,axis=0))  #数据标准化

'''模糊聚类算法'''
U,center,fcm_obj_fcn = FuzzyClustering.esfcm(data,cluster_n,label[1:140])
label_pred,abaaba = np.where(U==np.max(U,axis=0)) #最大值索引


'''画图'''
plt.plot(fcm_obj_fcn)
plt.show()

'''性能评价'''
label_pred = label_pred + 1     #因为索引是从零开始，但标签是从1开始
print(U[:,1])
print(accuracy_score(label.tolist(),label_pred.tolist()))