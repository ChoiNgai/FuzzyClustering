# 模糊聚类

**keywords: Fuzzy clustering，semi-supervised**

## introduce

本开源项目为模糊聚类算法python代码，主要算法包括：

- FCM（模糊C均值算法）
- MEC （极大熵模糊聚类算法）
- KFCM（核模糊聚类算法）
- SFCM （半监督模糊聚类算法）
- eSFCM （基于信息熵的半监督模糊聚类算法）
- SMUC （基于度量学习与信息熵的半监督模糊聚类算法）



以这些算法为基础的相关论文可参考本人的谷歌学术主页：[Wei Cai，Guangdong University of Technology](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=pYX8lisAAAAJ)



## project structure

-  dataset：数据集
- ClusterAidedComputing.py ：包括聚类常用的一些函数
- ClusteringIteration.py ：包括聚类算法迭代式
- test.py ： 测试脚本
- FuzzyClustering.py ：模糊聚类算法



算法都封装在FuzzyClustering.py里，FuzzyClustering.py调用ClusterAidedComputing.py和ClusteringIteration.py



## 算法调用

### 参数

（以下为所有模糊聚类算法都有的参数）

data ：数据集，统一使用数组（darry）

cluster_n ：类簇中心数 

max_iter ：最大迭代次数

e ：目标函数值变化最小阈值

printOn ：打印迭代情况开关（当printOn=1时打印迭代情况）

### 调用规则

所有的函数都需要输入data和cluster_n，其余参数可能有预设参数（若有预设参数则可以不输入，不输入则采用默认参数）

### 算法函数

- FCM

```python
U,V,obj_fcn = fcm(data,cluster_n,m = 2,max_iter = 1000,e = 0.00001,printOn = 1)
```

或

```python
U,V,obj_fcn = fcm(data,cluster_n)
```

如上，m ,max_iter,e ,printOn这四个参数已有默认参数，可不设置 

- MEC

```python
U,V,obj_fcn = mec(data,cluster_n,gamma=0.01,max_iter = 1000,e = 0.00001,printOn = 1)
```

gamma ：惩罚系数

- KFCM

  sigma ：高斯核标准差

  lamda ：惩罚系数

```python
kfcm(data,cluster_n,sigma=2,m=2,lamda=0.1,max_iter = 1000,e = 0.00001,printOn = 1)
```

- SFCM

```python
U,V,obj_fcn = sfcm(data,cluster_n,label,m = 2,max_iter = 1000,e = 0.00001,alpha=5,printOn = 1)
```

label ：标签（array格式）

- eSFCM

```python
U,V,obj_fcn = esfcm(data,cluster_n,label,max_iter = 1000,e = 0.00001,lamda=1,printOn = 1)
```

- SMUC

```smuc
U,V,obj_fcn = smuc(data,cluster_n,label,max_iter = 1000,e = 0.5,lamda=1,printOn = 1)
```

### 例

迭代目标函数值变化图：

![Figure_1](https://cdn.jsdelivr.net/gh/ChoiNgai/ImageServer/img/Figure_1.png)![image-20210325193158437](https://cdn.jsdelivr.net/gh/ChoiNgai/ImageServer/img/image-20210325193158437.png)