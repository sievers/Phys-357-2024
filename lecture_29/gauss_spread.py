import numpy as np
from scipy.linalg import eigh_tridiagonal
from matplotlib import pyplot as plt
plt.ion()


#set up potential for step function
x=np.linspace(-20,20,4001)
dx=x[1]-x[0]
V0=0
V=0.0+(x>5)*V0


#V[x>5.3]=0 #fun tunneling for 9.5 for initial gradient
#solve Schrodinger eigenproblem
Hdiag=1.0/dx+V
Hoff=-0.5*np.ones(len(x)-1)/dx
e,psi=eigh_tridiagonal(Hdiag,Hoff)

#start with a gaussian wave packet
myg=np.exp(-0.5*(x)**2/2**2)

#now we want to start it moving to the right.
#E=k^2/2 if we set hbar to 1, so we want k^2/2>V0
#
#now we'll add a phase ramp to start the particle moving
k=np.sqrt(2*V0)*0.0 #10 is about boundary to go to transmission regime
psi0=myg*np.exp(-1J*x*k)
amps0=psi.T@psi0

p0=np.abs(amps0)**2
E0=np.sum(p0*e)/np.sum(p0)
Esqr=np.sum(p0*(e**2))/np.sum(p0)
Esig=np.sqrt(Esqr-E0**2)
#print('starting energy is ',np.sum(np.abs(amps0)**2*e)/np.sum(np.abs(amps0)**2))
print('starting energy is ',E0,' +/- ',Esig)


for t in np.arange(0,2000,0.3):
    amps=amps0*np.exp(1J*t*e)
    psi_cur=psi@amps
    plt.clf()
    plt.plot(x,np.abs(psi_cur)**2)
    #plt.plot(x,V)
    plt.ylim([0,1])
    plt.pause(0.01)
