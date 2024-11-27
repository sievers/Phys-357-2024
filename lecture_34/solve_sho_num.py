import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def sho_fun(y,psi,eps):
    #psi'=(psi)'
    #psi''=(y^2-eps)psi
    return np.asarray([psi[1],(y**2-eps)*psi[0]])

def rk4(fun,x,y,h,eps):
    k1=fun(x,y,eps)*h
    k2=h*fun(x+h/2,y+k1/2,eps)
    k3=h*fun(x+h/2,y+k2/2,eps)
    k4=h*fun(x+h,y+k3,eps)
    dy=(k1+2*k2+2*k3+k4)/6
    return y+dy


y=np.linspace(0,7,501)
h=y[1]-y[0]
psi=np.zeros([len(y),2])
for eps in np.arange(6.9,7.1,0.0001):
    psi[0,:]=np.asarray([0,1]);even=False
    #psi[0,:]=np.asarray([1,0]);even=True
    for i in range(len(y)-1):
        psi[i+1,:]=rk4(sho_fun,y[i],psi[i,:],h,eps)
    plt.clf()
    plt.plot(y,psi[:,0],'k')
    if even:
        plt.plot(-y,psi[:,0],'k')
    else:
        plt.plot(-y,-psi[:,0],'k')
    plt.title(r'$\epsilon=$'+repr(eps))
    plt.pause(0.001)
