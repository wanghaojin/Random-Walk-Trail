import numpy as np

a = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
a = np.array(a,dtype = float)
b = a.copy()
for i in range(a.shape[0]):
    b[i] /= np.sum(a[i])
print(a)
print(b)