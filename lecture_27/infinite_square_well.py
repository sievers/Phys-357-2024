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
    modes[:,n]=np.sin((n+1)*np.pi*x/a) #these are the eigenstates of the ISW
    modes[:,n]=modes[:,n]/np.sqrt(np.sum(modes[:,n]**2)) #normalize
E=(np.arange(nmax)+1)**2*np.pi**2/a**2 #and these are the analytic energies

psi0=np.exp(-0.5*(x-a/2)**2/0.25**2) #pick a starting wave function
psi0=psi0-psi0[0]
psi0=psi0*np.exp(-4J*x)  #give it a nudge if you want

#find the amplitudes of our starting wave function in terms of the
#eigenstates.  normally we might have to least-squares, but we know
#the eigenstates are orthogonal, so modes.T@psi0 does what we want.
amps=modes.T@psi0 

#check that when we try to reconstruct our wave function
#we get something very close to what we started with.  If this
#number is not very small, you probably want to increase the number
#of states you are using for the basis vectors
psi_pred=modes@amps
print('starting accuracy is ',np.std(psi0-psi_pred))
for t in np.arange(0,50,0.001):
    #this is the time evolution of the state amplitudes
    amps_cur=amps*np.exp(-1J*E*t)
    #and our current wave function is sum of current amplitudes
    #times eigenstates
    psi_cur=modes@amps_cur
    #and plot what our wave function did!
    plt.clf()
    plt.plot(x,np.abs(psi_cur)**2)
    plt.ylim(0,1)
    plt.pause(0.01)
