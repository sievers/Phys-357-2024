import numpy as np
from matplotlib import pyplot as plt
plt.ion()
#code to look at infinite square well, where
#we have dialed in the wave functions because we know them
a=3.0
x=np.linspace(0,a,3001)
nmax=200
modes=np.zeros([len(x),nmax])
for n in range(nmax):
    modes[:,n]=np.sin((n+1)*np.pi*x/a)
    modes[:,n]=modes[:,n]/np.sqrt(np.sum(modes[:,n]**2))
E=(np.arange(nmax)+1)**2*np.pi**2/a**2

psi0=np.exp(-0.5*(x-a/2)**2/0.25**2)
psi0=psi0-psi0[0]
psi0=psi0*np.exp(4J*x)
amps=modes.T@psi0
psi_pred=modes@amps
for t in np.arange(0,50,0.001):
    amps_cur=amps*np.exp(-1J*E*t)
    psi_cur=modes@amps_cur
    plt.clf()
    plt.plot(x,np.abs(psi_cur)**2)
    plt.ylim(0,1)
    plt.pause(0.01)
