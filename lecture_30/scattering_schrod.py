import numpy as np
#from scipy.linalg import eigh_tridiagonal
import schrod
from matplotlib import pyplot as plt
plt.ion()


#set up potential for step function
xmax=80
x=np.linspace(-80,80,6001)
V0=1
x0=xmax/4
V=0.0+(x>x0)*V0
states=schrod.States(x,V)


#start with a gaussian wave packet
sig=0.1*xmax
myg=np.exp(-0.5*(x+x0)**2/sig**2)

#now we'll add a phase ramp to start the particle moving
k=np.sqrt(2*V0)*1.0
psi0=myg*np.exp(1J*x*k)

E0,Esig=states.get_energy(psi0)
print('starting energy/uncertainty: ',E0,Esig)

for t in np.arange(0,2000,0.2):
    psi=states.evolve(psi0,t)
    plt.clf()
    plt.plot(x,np.abs(psi)**2)
    plt.plot(x,V)
    plt.pause(0.01)
