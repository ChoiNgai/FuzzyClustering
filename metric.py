import numpy
x = numpy.array([[3,4],[5,6],[2,2],[8,4]])
xT=x.T
print(xT)
D=numpy.cov(xT)
print(D)
S=numpy.linalg.inv(D)
print(S)
tp=x[0]-x[1]
print(tp)
print(numpy.sqrt(numpy.dot(numpy.dot(tp,S),tp.T)))

import numpy as np
x = np.array([[1,2,3],[2,4,3]])
D = np.cov(x)
invD = np.linalg.inv(D)
tp = x.T[0]-x.T[1]
dist = np.sqrt(np.dot(np.dot(tp,invD),tp.T))