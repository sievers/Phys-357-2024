import numpy as np
from scipy.linalg import eigh_tridiagonal

class States:
    def __init__(self,x,V):
        #solve the Schrodinger equation for the stationary
        #states of a given potential.  We'll assume m=hbar=1
        #so please keep that in mind when using this to get
        #physical results.
        dx=x[1]-x[0]
        self.dx=dx
        n=len(x)
        H_diag=V+np.ones(n)/dx**2
        H_off=-0.5*np.ones(n-1)/dx**2
        E,psi=eigh_tridiagonal(H_diag,H_off)
        self.E=E
        self.psi=psi
    def get_energy(self,psi):
        #return the energy and uncertainty in energy for state Psi
        amps=np.conj(self.psi).T@psi
        probs=np.abs(amps)**2
        probs=probs/np.sum(probs) #make sure we're normalized
        E=np.sum(probs*self.E)
        Esqr=np.sum(probs*(self.E**2))
        Esig=np.sqrt(Esqr-E**2)
        return E,Esig
    def evolve(self,psi,t):
        #evolve a starting wave function psi forward in time
        #by time t.  
        amps_org=np.conj(self.psi).T@psi
        phases=-t*self.E
        amps=np.exp(1J*phases)*amps_org
        return self.psi@amps
