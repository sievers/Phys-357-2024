#code to show that the normalization of of the FT/IFT
#pair is really 2pi.  

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

#set up a grid in x, calculate dx
x=np.linspace(-1,1,3001)
dx=x[2]-x[1]

#now set up a grid in k, calculate dk
fac=4
k=np.linspace(fac*x[0]/dx,fac*x[-1]/dx,int(fac*len(x)))
dk=k[2]-k[1]

#put in some function.  Ideally something that's small-ish at
#the edges of the boundary to avoid ringing.  It'll still sort-of
#work if you don't, but errors will be larger/you'll want to tune
#your k grid more.

y=np.exp(-0.5*x**2/(0.25**2))*np.sqrt(1+np.sin(x))
#y=0*x;y[np.abs(x)<0.01]=1

#take the forward transform.  brute force, and the
#integral is roughly the sum of f(x)exp(-ikx) time dx
F=(0*k).astype('complex64')
for i in range(len(k)):
    F[i]=np.sum(y*np.exp(-1J*k[i]*x))*dx

#now take the inverse transform, once again brute-forcing
f=(0*y).astype('complex64')
for i in range(len(x)):
    f[i]=np.sum(F*np.exp(1j*k*x[i]))*dk

f=f/(2*np.pi) #after applying this normalization, we expect f to match y
print('round-trip scatter: ',np.std(f-y))
