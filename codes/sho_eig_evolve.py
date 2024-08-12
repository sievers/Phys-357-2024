import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh_tridiagonal
import time
plt.ion()

#let's set up our x-range first
x=np.linspace(-15,15,2001)
dx=np.median(np.diff(x))

#this sets up our second derivative oeprator
vec=np.zeros(len(x))
vec[0]=2
vec[1]=-1
mat=toeplitz(vec/dx**2)

#set up our potential
V=0.25*x**2

#and make our Hamiltonian
H=mat+0.25*np.diag(V)


#solve the Shrodinger equation (in one line!)
e,v=np.linalg.eigh(H)

#we could alternatively use the specialized symmetric tridiagonal
#eigenvalue routine in scipy.  This is would be much faster and take less memory!
#e,v=eigh_tridiagonal(2*np.ones(len(x))/dx**2+V,-np.ones(len(x)-1)/dx**2)


#so for something interesting, let's pretend we suddenly shifted our potential
#which is equivalent to shifting the particle.  Then we can watch the time
#evolution of the particle oscillating around in the potential
#by picking the 0'th eigenvector, we're starting with the particle in the
#ground state before moving the potential

shift=4
dn=int(shift/dx)
psi0=np.roll(v[:,0],dn)

#get the shifted state in terms of the eigenfunctions
psi0_eig=v.T@psi0

print('starting energy is :',np.sum(np.abs(psi0_eig)*e))
dt=0.01/e[0]
for i in range(300):
    psi=v@(np.exp(1J*e*dt*i)*psi0_eig)
    pdf=np.abs(psi)**2
    plt.clf()
    plt.plot(x,pdf/dx)
    plt.title('SHO Potential')
    plt.show()
    plt.pause(0.01)



barrier_height=20
barrier_width=0.5
V_barrier=V+barrier_height*np.exp(-0.5*x**2/barrier_width**2)
e2,v2=eigh_tridiagonal(2*np.ones(len(x))/dx**2+0.25*V_barrier,-np.ones(len(x)-1)/dx**2) 



psi0_eig=v2.T@psi0
print('starting energy w/barrier is :',np.sum(np.abs(psi0_eig)*e2))
for i in range(3600):
    psi=v2@(np.exp(1J*e2*dt*i)*psi0_eig)
    pdf=np.abs(psi)**2
    plt.clf()
    plt.plot(x,pdf/dx)
    plt.title('SHO Potential w/Barrier')
    plt.plot(x,V_barrier/V_barrier.max())
    plt.show()

    plt.ylim([0,0.7])
    plt.pause(0.01)
    if i==0:
        plt.pause(3)


