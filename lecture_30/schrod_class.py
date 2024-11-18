import numpy as np
from matplotlib import pyplot as plt
import schrod
plt.ion()

xmax=80
x=np.linspace(-xmax,xmax,6001)
x0=xmax/4
V0=1.0
V=0*x+V0*(x>x0)
states=schrod.States(x,V)

sig=xmax/10
psi_gauss=np.exp(-0.5*(x+x0)**2/sig**2)
k_crit=np.sqrt(2*V0)
k=1.0*k_crit
psi0=psi_gauss*np.exp(1J*k*x)

for t in np.arange(0,2000,1.3):
    psi=states.evolve(psi0,t)
    plt.clf()
    plt.plot(x,np.abs(psi)**2)
    plt.plot(x,V)
    plt.pause(0.001)

#plt.clf()
#plt.plot(x,V)
#plt.plot(x,psi_gauss)
#plt.show()
