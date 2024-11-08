import numpy as np
from scipy.linalg import eigh_tridiagonal
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-2,2,3000)
dx=x[2]-x[1]
vec=np.ones(len(x))/dx**2 #I'll get my second derivative from this

V0=5000
V=0*x
a=1.0
V[np.abs(x)>a/2]=V0

e,psi=eigh_tridiagonal(vec+V,-0.5*vec[:-1])
plt.figure(1)
plt.clf()
plt.plot(x,V/V0)
plt.plot(x,psi[:,:3]/np.sqrt(dx))
plt.show()

epred=np.pi**2*(np.arange(len(x))+1)**2/(2*a**2)
print('first few measured energies: ')
print(e[:5])
print('first few predicted energies: ')
print(epred[:5])

psi0=0*x
tmp=psi[::2,0] #take every second point to squish by factor of 2
m=(len(x)-len(tmp))//2
psi0[m:len(tmp)+m]=tmp
psi0=psi0*np.exp(0J*x)
amps=psi.T@psi0
tvec=np.arange(0,50,0.001)
plt.figure(2)
for i in range(len(tvec)):
    amps_use=amps*np.exp(-1J*tvec[i]*e)
    psi_cur=psi.T@amps_use
    plt.clf()
    plt.plot(x,np.abs(psi_cur**2)/dx)
    plt.ylim(0,1)
    plt.pause(0.02)

