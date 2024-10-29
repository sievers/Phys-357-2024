#Code to numerically evaluate the evolution of spin
#in magnetic resonance.  We want to evolve the system
#    hbar/2 ( om_0            om_1 cos(om t)  ) a(t) =  i hbar ( da/dt )
#           ( om_1 cos(om t)  -om_0           ) b(t)           ( db/dt )
# om_0 is the resonance frequency of an applied constant field,
# om_1 is the resonance frequency of an applied oscillating field,
# and om is the frequency of oscillation of that applied field.
# in general, om_0 << om_1
#
# we'll also use this code to look at what happens when the modulated field
# is in the z direction, which makes the system of equations
# \hbar/2 ( om_0 + om_1 cos(om t)  0   ) a(t)  = i hbar (da/dt)
#         (  0   -(om_0+om_1 cos(om t) ) b(t)           (db/dt)

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def fun(t,psi,om_0,om_1,om):
    #return d Psi/dt, which we get from taking -i/2 []Psi, where [] is the
    #matrix above, and the numerical factor comes from solving for d Psi/dt
    #(which is the vector of (da/dt,db/dt))
    #this Psi is for standard magnetic resonance, where there's an S_x term
    #driven sinusoidally
    mat=np.asarray([[om_0,om_1*np.cos(om*t)],[om_1*np.cos(om*t),-om_0]])
    return -0.5J*mat@psi

def fun2(t,psi,om_0,om_1,om):
    #return d Psi/dt not for a purely z-direction field, but still with
    #the same time-varying forcing.  The behavior is very different if the
    #time-varying component is the same direction as the static component.
    mat=np.asarray([[om_0+om_1*np.cos(om*t),0],[0,-om_0-om_1*np.cos(om*t)]])
    return -0.5J*mat@psi


def rk4(f,t,psi,om_0,om_1,om,dt):
    #take a Runga-Kutta 4th order step of stepsize dt
    k1=f(t,psi,om_0,om_1,om)
    k2=f(t+dt/2,psi+dt/2*k1,om_0,om_1,om)
    k3=f(t+dt/2,psi+dt/2*k2,om_0,om_1,om)
    k4=f(t+dt,psi+dt*k3,om_0,om_1,om)
    return psi+dt*(k1+2*k2+2*k3+k4)/6

psi_0=np.asarray([1,0])
om_0=1.0
om_1=0.1
om=1.1

#set up and solve the time-evolution problem
t=np.arange(0,100,0.01)
psi=np.zeros([len(t),2],dtype='complex')
psi[0,:]=psi_0
for i in range(len(t)-1):
    dt=t[i+1]-t[i]
    psi[i+1,:]=rk4(fun,t[i],psi[i,:],om_0,om_1,om,dt)

plt.clf()
plt.plot(t,np.abs(psi)**2)
plt.show()
